import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
from planning import Proposition, State
from representation import REPRESENTATIONS, Representation
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from abc import ABC, abstractmethod
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU, Dropout, LeakyReLU, BatchNorm1d
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import Tensor
from typing import Optional, List, FrozenSet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.nn.conv import (
#     RGCNConv,
#     FastRGCNConv,
# )  # (slow and/or mem inefficient)

""" This file contains two classes:
    1. a class for an actual GNN object
    2. a class which acts a heuristic function object and contains a GNN and Representation object.
"""


def construct_mlp(in_features: int, out_features: int, n_hid: int) -> torch.nn.Module:
    return Sequential(
        Linear(in_features, n_hid),
        ReLU(),
        Linear(n_hid, out_features),
    )


class RGNNLayer(Module):
    """单个RGNN层"""
    def __init__(self, in_features: int, out_features: int, n_edge_labels: int, aggr: str):
        super(RGNNLayer, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(n_edge_labels):  # 对于每类边: 分别创建线性卷积层
            self.convs.append(LinearConv(in_features, out_features, aggr=aggr).jittable()) # jittable用于加速
        self.root = Linear(in_features, out_features, bias=True)
        return

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor]) -> Tensor:
        """更新节点特征
        输入: 整张图的节点特征, 邻接矩阵列表(按类别)
        输出: 更新后的图
        """
        x_out = self.root(x)    # 先对更新节点特征进行线性变换(保证维度一致)
        for i, conv in enumerate(self.convs):  # bottleneck; difficult to parallelise efficiently
            x_out += conv(x, list_of_edge_index[i]) # 根据边类别: 分别对与该点相邻的点进行卷积聚合, 提取特征后相加
        return x_out


class LinearConv(MessagePassing):
    """线性卷积层, 用于聚合邻居节点特征"""
    propagate_type = {"x": Tensor}

    def __init__(self, in_features: int, out_features: int, aggr: str) -> None:
        super().__init__(aggr=aggr)
        self.f = Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """消息传播
        输入: 节点特征(整张图), 邻边索引
        输出: 整张图待传播的消息
        """
        # propagate_type = {'x': Tensor }
        x = self.f(x)   # 对节点特征简单线性变换
        # propagate = message + aggregate + update, 其中message传递邻点特征(根据index索引), aggregate根据输入选择max或mean, update更新节点特征
        x = self.propagate(edge_index=edge_index, x=x, size=None)
        return x


class RGNN(nn.Module):
    """
    The class can be compiled with jit or the new pytorch-2. However, pytorch-geometric
    has yet to provide compiling for GNNs with variable sized graph inputs.
    RGNN模块, 用于对图编码处理解码, 得到启发值
    """

    def __init__(self, params) -> None:
        super().__init__()
        # 相关参数
        self.in_feat = params["in_feat"]                # 输入特征维度(由图表示决定)
        self.out_feat = params["out_feat"]              # 输出特征维度 默认为1
        self.nhid = params["nhid"]                      # 隐藏层特征维度
        self.aggr = params["aggr"]                      # 聚合方式: max, mean, sum
        self.n_edge_labels = params["n_edge_labels"]    # 边的类别数(不同类型的边)
        self.nlayers = params["nlayers"]                # RGNN层数(对应L)
        self.rep_type = params["rep"]                   # 图表示方法
        self.rep = None
        self.device = None
        self.batch = False

        # 全局池化方法
        if params["pool"] == "max":
            self.pool = global_max_pool
        elif params["pool"] == "mean":
            self.pool = global_mean_pool
        elif params["pool"] == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError

        self.initialise_layers()    # 初始化层

        return

    @abstractmethod
    def create_layer(self) -> None:
        raise NotImplementedError

    def initialise_layers(self) -> None:
        """
        RGNN相关层结构
        emb: 编码层, 将输入特征线性变换到隐藏层维度
        layers: RGNN消息传递层, 特征维度不变
        mlp_h: 启发式解码层, 将隐藏层特征通过MLP解码到输出维度
        """
        self.emb = torch.nn.Linear(self.in_feat, self.nhid)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.nlayers):
            self.layers.append(self.create_layer())
        self.mlp_h = construct_mlp(in_features=self.nhid, n_hid=self.nhid, out_features=self.out_feat)
        return

    def create_layer(self):
        return RGNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels, aggr=self.aggr)

    def node_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        # 对节点特征进行编码及更新方法: 先编码, 然后每次用RGNN更新后ReLU激活
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, list_of_edge_index)
            x = F.relu(x)
        return x

    def graph_embedding(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        # 图编码及更新方法: 节点更新, 然后全局池化得到单一向量(维度为隐藏层维度)
        x = self.node_embedding(x, list_of_edge_index, batch)
        x = self.pool(x, batch)
        return x

    def forward(
        self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]
    ) -> Tensor:
        """overwrite typing (same semantics, different typing) for jit"""
        x = self.graph_embedding(x, list_of_edge_index, batch)  # 对图进行编码, 更新, 池化
        h = self.mlp_h(x)   # 对全局特征利用MLP解码得到单一启发值
        h = h.squeeze(1)    # 去掉多余的维度
        return h

    def name(self) -> str:
        return type(self).__name__


class Model(nn.Module):
    """
    A wrapper for a GNN which contains the GNN, additional informations beyond hyperparameters,
    and helpful methods such as I/O and providing an interface for planners to call as a heuristic
    evaluator.
    封装RGNN, 可用直接作为状态的启发式函数使用
    同时定义相关有用函数, 方便查看信息、更改设置等 
    """

    def __init__(self, params=None, jit=False) -> None:
        super().__init__()
        if params is not None:
            self.model = None
            self.jit = jit
            self.rep_type = params["rep"]
            self.rep = None
            self.device = None
            self.batch = False
            self.create_model(params)   # 创建RGNN模型
        if self.jit:
            self.model = torch.jit.script(self.model)   # 编译模型, 加速推理
        return
    
    def set_eval(self) -> None:
        self.model.eval()
        return

    def lifted_state_input(self) -> bool:
        return self.rep.lifted

    def dump_model_stats(self) -> None:
        print(f"Model name: RGNN")
        print(f"Device:", self.device)
        print(f"Number of parameters:", self.get_num_parameters())
        print(f"Number of layers:", self.model.nlayers)
        print(f"Number of hidden units:", self.model.nhid)
        return

    def load_state_dict_into_gnn(self, model_state_dict) -> None:
        """Load saved weights"""
        self.model.load_state_dict(model_state_dict)

    def forward(self, data):
        return self.model.forward(data.x, data.edge_index, data.batch)

    def embeddings(self, data):
        return self.model.graph_embedding(data.x, data.edge_index, data.batch) 
    
    def forward_to_embeddings(self, data):
        """添加图编码"""
        return self.model.graph_embedding(data.x, data.edge_index, data.batch)

    def forward_from_embeddings(self, embeddings):
        """从图编码得到启发值"""
        x = self.model.mlp_h(embeddings)
        # x = x.squeeze(1)
        x = x.squeeze(1)    # 去掉多余的维度
        return x

    def initialise_readout(self):
        if self.jit:
            self.model.mlp = torch.jit.script(
                construct_mlp(
                    in_features=self.model.nhid,
                    n_hid=self.model.nhid,
                    out_features=self.model.out_feat,
                )
            )
        else:
            self.model.mlp = construct_mlp(
                in_features=self.model.nhid,
                n_hid=self.model.nhid,
                out_features=self.model.out_feat,
            )
        return

    def update_representation(self, domain_pddl: str, problem_pddl: str, args, device):
        self.rep: Representation = REPRESENTATIONS[self.rep_type](domain_pddl, problem_pddl)
        self.rep.convert_to_pyg()
        self.device = device
        return

    def update_device(self, device):
        self.device = device
        return

    def batch_search(self, batch: bool):
        self.batch = batch
        return

    def print_weights(self) -> None:
        weights = self.state_dict()
        for weight_group in weights:
            print(weight_group)
            print(weights[weight_group])
        return

    def get_num_parameters(self) -> int:
        """Count number of weight parameters"""
        # https://stackoverflow.com/a/62764464/13531424
        # e.g. to deal with case of sharing layers
        params = sum(
            dict((p.data_ptr(), p.numel()) for p in self.parameters() if p.requires_grad).values()
        )
        # params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return params

    def get_num_zero_parameters(self) -> int:
        """Count number of parameters that are zero after training"""
        zero_weights = 0
        for p in self.parameters():
            if p.requires_grad:
                zero_weights += torch.sum(torch.isclose(p.data, torch.zeros_like(p.data)))
        return zero_weights

    def print_num_parameters(self) -> None:
        print(f"number of parameters: {self.get_num_parameters()}")
        return

    def set_zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def create_model(self, params):
        """创建RGNN模型"""
        self.model = RGNN(params)

    def h(self, state: State) -> float:
        """计算状态启发值: 状态 -> 图表示 -> 启发值"""
        with torch.no_grad():
            x, edge_index = self.rep.state_to_tensor(state) # 状态转化为图, 再转化为特征
            x = x.to(self.device)
            for i in range(len(edge_index)):
                edge_index[i] = edge_index[i].to(self.device)
            h = self.model.forward(x, edge_index, None)
            h = round(h.item()) # 启发值取整
            
            return h

    def h_batch(self, states: List[State]) -> List[float]:
        """批量计算启发值"""
        with torch.no_grad():
            data_list = []
            for state in states:
                x, edge_index = self.rep.state_to_tensor(state)
                data_list.append(Data(x=x, edge_index=edge_index))
            loader = DataLoader(dataset=data_list, batch_size=min(len(data_list), 32))
            hs_all = []
            # 数据已转化为多个batch, 每个batch并行处理
            for data in loader:
                data = data.to(self.device)
                hs = self.model.forward(data.x, data.edge_index, data.batch)
                hs = hs.detach().cpu().numpy()  # annoying error with jit
                hs_all.append(hs)
            hs_all = np.concatenate(hs_all)
            hs_all = np.rint(hs_all)
            hs_all = hs_all.astype(int).tolist()
            return hs_all

    def __call__(self, node_or_list_nodes):  # call on Pyperplan search
        """用于pyperplan搜索"""
        if self.batch:
            states = [n.state for n in node_or_list_nodes]
            h = self.h_batch(states)  # list of states
        else:
            state = node_or_list_nodes.state
            h = self.h(state)  # single state
        return h

    def name(self) -> str:
        return self.model.name()
    
