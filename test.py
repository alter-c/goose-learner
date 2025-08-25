import networkx as nx
import numpy as np
import time
import os
import argparse
from torch_geometric.data import Data

import representation
from representation import *
from representation.deepwalk import *
from dataset.dataset import get_plan_info
from dataset.goose_domain_info import GOOSE_DOMAINS


""" 修改图特征

###  原流程: 根据domain和problem构建networkx图 
# -> 将图转化为torch数据(convert_to_pyg) 
# -> 处理实例状态(str_to_state) 
# -> 将实例状态添加到相关节点特征和边标签(state_to_tensor)
# -> 构建torch数据格式 Data(x=x, edge_index=edge_index, y=y)

### 目标流程: 根据domain和problem构建networkx图 
# -> 处理实例状态(str_to_state) 
# -> 将实例状态添加到图中(state_to_graph)
# -> 利用deepwalk提取图特征X
# -> 将图转化为torch数据(convert_to_pyg)
# -> 将特征转为torch格式, 提取边索引(edge_indices)
"""

DATASET_DIR = "../dataset/goose"


def draw_graph(G):
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.use('TkAgg')
    nx.draw(G, with_labels=True)
    plt.axis('on')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=GOOSE_DOMAINS)
    # data arguments
    parser.add_argument(
        "-r",
        "--rep",
        type=str,
        default="llg",
        choices=REPRESENTATIONS_STR,
    )
    parser.add_argument(
        "-p",
        "--planner",
        type=str,
        default="fd",
        choices=["fd", "pwl"]
    )
    args = parser.parse_args()
    return args

def get_files(args):
    domain = args.domain
    domain_pddl = f"{DATASET_DIR}/{domain}/domain.pddl"
    problem_dir = f"{DATASET_DIR}/{domain}/train"
    plan_dir = f"{DATASET_DIR}/{domain}/train_solution"

    problem_list = sorted(list(os.listdir(problem_dir)))
    problem_pddl = problem_list[0]
    plan_file = problem_pddl.replace('.pddl', '.plan')

    # 更新为绝对路径
    problem_pddl = f"{problem_dir}/{problem_pddl}"
    plan_file = f"{plan_dir}/{plan_file}"
    print(f"测试文件路径\ndomain: {domain_pddl}\nproblem: {problem_pddl}\nplan: {plan_file}")

    return domain_pddl, problem_pddl, plan_file



if __name__ == "__main__":
    args = parse_args()
    domain_pddl, problem_pddl, plan_file = get_files(args)
    plan = get_plan_info(domain_pddl, problem_pddl, plan_file, args)

    rep_str = "llg"
    rep = representation.REPRESENTATIONS[rep_str](domain_pddl, problem_pddl) 
    rep.convert_to_pyg()
    print(f"初始图特征大小{rep.x.shape}")

    graphs = []
    total_time = 0
    # goal_x, _ = rep.goal_to_tensor()
    # print(f"目标特征大小: {goal_x.shape}")
    
    for state, schema_cnt in plan:
        s = time.time()
        state = rep.str_to_state(state)
        X, edge_indices = rep.state_to_tensor(state)
        y = sum(schema_cnt.values())
        graph = Data(x=X, edge_index=edge_indices, y=y)
        graphs.append(graph)
        e = time.time()
        total_time += e - s
        print(f"状态特征大小: {X.shape}")
    
    print(graphs[0].x.shape)

    print(f"总耗时: {total_time}秒")
    print(f"图特征大小: {graphs[0].x.shape}")
    print(f"数据集大小: {len(graphs)}")


    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader
    import torch

    from gnns.loss import BCELoss, MSELoss, CombinedLoss
    from gnns.gnn import Model

    from gnns.train_eval import train, evaluate

    trainset, valset = train_test_split(graphs, test_size=0.15, random_state=4550)

    batch_size = 32
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
    )
    in_feat = train_loader.dataset[0].x.shape[1]

    model_params = {
        "in_feat": in_feat,
        "out_feat": 1,
        "nlayers": 4,
        "n_edge_labels": 6,
        "nhid": 64,
        "aggr": "mean",
        "pool": "sum",
        "rep": "llg",
    }
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(params=model_params).to(device)

    lr = 0.001
    reduction = 0.1
    patience = 10
    epochs = 10

    criterion = MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", verbose=True, factor=reduction, patience=patience
    )
    for e in range(epochs):
        train_stats = train(model, device, train_loader, criterion, optimiser)
        print(train_stats)
