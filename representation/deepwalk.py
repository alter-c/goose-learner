"""用于为图生成deepwalk特征"""

import os
import random
import time
import copy
import numpy as np
from gensim.models import Word2Vec
from concurrent.futures import ThreadPoolExecutor, as_completed

DEEPWALK_FEATURE_SIZE = 8  # deepwalk特征维度

_ROOT_DIR = os.environ.get('GOOSE_ROOT')
_MODEL_DIR = os.path.join(_ROOT_DIR, "experiments/models")
os.makedirs(_MODEL_DIR, exist_ok=True)


def random_walk(G, node, path_length):
    """输入起始节点和路径长度, 生成单条随机游走节点序列"""
    walk = [node]
    for i in range(path_length - 1):
        # 获取当前节点的邻居节点
        temp = list(G.neighbors(walk[-1]))
        if len(temp) == 0: # 如果无邻点，结束随机游走
            break
        walk.append(random.choice(temp))

    # 将节点转化为字符串, 否则在word2vec中查询词向量时tuple中元素会当作不同词递归查询
    str_walk = [str(node) for node in walk] 
    return str_walk

def create_walk_data(G, new_nodes=None):
    """生成随机游走数据"""
    # 根据更新模式设置游走节点
    node_list = G.nodes() if new_nodes is None else new_nodes 
    gamma = 5 # 每个节点生成的随机游走序列数量
    walk_length = 10 # 每次随机游走的长度
    random_walks = []

    # def node_walks(node): # 并行生成的时间优势并不明显
    #     return [random_walk(G, node, walk_length) for _ in range(gamma)]
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(node_walks, node) for node in node_list]
    #     for future in as_completed(futures):
    #         random_walks.extend(future.result())
    
    for _ in range(gamma): 
        for node in node_list:
            random_walks.append(random_walk(G, node, walk_length))

    return random_walks

def deepwalk(G):
    """
    deepwalk算法简单实现
    生成deepwalk初始模型
    """
    random_walks = create_walk_data(G)

    ### 训练Node2Vec模型
    model = Word2Vec(vector_size=DEEPWALK_FEATURE_SIZE,
                    window=4,
                    sg=1,
                    hs=1,
                    seed=3407
                    )
    model.build_vocab(random_walks, progress_per=2)
    model.train(random_walks, total_examples=model.corpus_count, epochs=10, report_delay=1)
    # print(model.wv.index_to_key)
    return model

def incremental_deepwalk(G, new_nodes, init_model: Word2Vec):
    """增量式更新deepwalk特征"""
    random_walks = create_walk_data(G, new_nodes)

    ### 训练Node2Vec模型
    # print("开始更新Node2Vec模型...")
    model = copy.deepcopy(init_model)
    model.build_vocab(random_walks, update=True, progress_per=2)
    model.train(random_walks, total_examples=model.corpus_count, epochs=2, report_delay=1)
    # print("完成Node2Vec模型更新, 耗时", e - s, "秒")
    return model

def load_deepwalk_model(G, model_name):
    """加载或生成deepwalk模型"""
    MODEL_FILE = os.path.join(_MODEL_DIR, f"deepwalk_{model_name}.model")
    if os.path.exists(MODEL_FILE):
        domain_model = Word2Vec.load(MODEL_FILE)
    else:
        print("generating base deepwalk model...")
        domain_model = deepwalk(G)
        domain_model.save(MODEL_FILE) 

    # 检查deepwalk特征维度是否匹配
    if domain_model.wv.vectors.shape[1] != DEEPWALK_FEATURE_SIZE:
        print("deepwalk feature size mismatch, re-generating model...")
        domain_model = deepwalk(G)
        domain_model.save(MODEL_FILE)

    return domain_model

# X = LSM(X, self.domain_X) # coverage and train loss is not good
from threadpoolctl import threadpool_limits
def LSM(X, domain_X):
    """
    Transform state features with LSM so that state's domain part features are fixed
    Params:
        X:          state feature            (n_state_nodes, m_features)
        domain_X:   standard domain feature  (n_domain_nodes, m_features)
    Returns:
        C: (in_features, out_features)
    """
    # numpy will use all available threads, thus effects cpu time 
    with threadpool_limits(limits=1):   # limit number of threads
        # get domain part features
        x = X[:len(domain_X)]
        y = domain_X
        # solve matrix C such that y≈xC, i.e., minimizes |y-xC|^2.(LSM)
        reg = np.eye(x.shape[1], dtype=np.float32) * 1e-6 # numpy: float64, pytorch: float32
        xtx_inv = np.linalg.inv(x.T @ x + reg)  # regularization 
        C = xtx_inv @ x.T @ y
        # standardize feature
        X = X @ C
    return X
    