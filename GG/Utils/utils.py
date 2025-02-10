from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from functools import reduce
from hashlib import md5
import tiktoken

def calculate_metrics(output, label):
    """
    计算F1-score和准确率
    :param output: 模型输出，形状为[batchsize, N, 1]，包含概率值
    :param label: 真实标签，形状为[batchsize, N, 1]，包含0和1的值
    :return: F1-score和准确率
    """
    # 将输出转换为二进制标签
    pred = (output > 0.5).float()

    # 将张量转换为numpy数组
    pred_np = pred.cpu().numpy().flatten()
    label_np = label.cpu().numpy().flatten()

    # 计算F1-score和准确率
    f1 = f1_score(label_np, pred_np)
    accuracy = accuracy_score(label_np, pred_np)

    return f1, accuracy

def calculate_metrics_chunk(y_pred, y_true):
    mae = torch.mean(torch.abs(y_true - y_pred))
    mse = torch.mean((y_true - y_pred) ** 2)

    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / (ss_total + 1e-8))  # 避免除零
    
    return mse, mae, r2

def coo_matrix_to_nx_graph(adj):
    N = adj.shape[0]
    vertex = np.arange(N)

    graph = nx.Graph()
    graph.add_nodes_from(vertex)

    adj = adj.coalesce()
    rows = adj.indices()[0]
    cols = adj.indices()[1]

    num = len(rows)
    for i in range(num):
        graph.add_edge(vertex[int(rows[i])], vertex[int(cols[i])])
        graph.add_edge(vertex[int(cols[i])], vertex[int(rows[i])])

    return graph

def calculate_query_distance(graph, cur_in, exist_dis):
    N = cur_in.shape[1]
    query_num = cur_in.shape[0]
    dis = nx.shortest_path_length(graph)
    dis = list(map(lambda x: x[1], dis))

    # cur_in shape: # (137, 348)
    # cur_in_dis 查询点 对于各个顶点的最短距离矩阵
    # query_num * N
    cur_in_dis = torch.zeros(cur_in.shape, dtype=torch.float32)
    for i in range(query_num):
        if torch.all(cur_in[i] == 0):
            continue
        source = []
        js = torch.where(cur_in[i] == 1)[0]
        for j in js:
            # 查询点: 把查询点到其他所有点的最短路径array加入到source中
            # shape: 1 * N
            if j in exist_dis:
                all_dis = exist_dis[j].to_dense()
            else:
                all_dis = torch.zeros(N, dtype=torch.float32)
                target = dis[j]
                for target_point, value in target.items():
                    all_dis[target_point] = value
                exist_dis[j] = all_dis.to_sparse()

            all_dis[all_dis == 0] = N
            all_dis[j] = 0
            all_dis = all_dis.unsqueeze(0)
            source.append(all_dis)
        # 按行拼接：可能有多个查询点
        try:
            source = torch.cat(source, dim=0)
        except:
            import pdb
            pdb.set_trace()
        assert source.shape[0] == torch.sum(cur_in[i])
        # 遍历所有节点
        # for j in range(N):
        # 多个查询点的最短距离：一次查询可能有多个query node，abc，将其对所有点的最短距离拼接，然后选对同一个点的的最小者
        cur_in_dis[i] = torch.min(source, dim=0).values

    ## normalization ###
    # 遍历每一个查询
    for j in range(query_num):
        # print(j)
        if torch.all(cur_in[j] == 0):
            continue
        # 按行normalization：查询点对于各顶点的最短路径距离的最大值A.shape[0]，表示不可达
        # 不可达的顶点的值设为0，其余的保持相同，然后选出最大值
        max_val = torch.max(cur_in_dis[j] * (cur_in_dis[j] < N))
        # print(cur_in_dis[j])
        # assert max_val > 0., (max_val)
        # 遍历每一个顶点
        flag = cur_in_dis[j] == N
        cur_in_dis[j][flag] = 0
        if max_val:
            cur_in_dis[j][~flag] = 1. - cur_in_dis[j][~flag] / (max_val + 1)

        # 查询点的cur_in_dis 设置为1
        flag = cur_in[j] == 1
        cur_in_dis[j][flag] = 1

    return cur_in_dis

def get_one_hot(indices, node_nums):
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    one_hot_tensor = F.one_hot(indices_tensor, num_classes=node_nums)
    one_hot_vector = one_hot_tensor.sum(dim=0)
    return one_hot_vector.unsqueeze(1).to(torch.float32)

def get_rows_or(indices, x):
    rows = x[indices]
    operations = reduce(torch.bitwise_or, rows)
    return operations.unsqueeze(1).to(torch.float32)

def get_normalized_laplacian(adj):
    # return A' =  D^-1/2 * A * D^-1/2
    if not adj.is_sparse:
        raise ValueError("The adjacency matrix should be a sparse tensor")
    deg = torch.sparse.sum(adj, dim=1).to_dense()

    # 计算度矩阵的逆平方根
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 构建度矩阵的逆平方根对角矩阵
    deg_inv_sqrt_mat = torch.sparse_coo_tensor(
        torch.stack([torch.arange(deg.size(0)), torch.arange(deg.size(0))]),
        deg_inv_sqrt,
        size=adj.size()
    )
    # 计算归一化的拉普拉斯矩阵
    normalized_laplacian = torch.sparse.mm(deg_inv_sqrt_mat, adj.to(torch.float32))
    normalized_laplacian = torch.sparse.mm(normalized_laplacian, deg_inv_sqrt_mat)


    return normalized_laplacian

def get_row_normalize(mx):
    """Row-normalize dense matrix"""
    rowsum = mx.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1  
    mx = mx / rowsum
    return mx


SYSTEM_PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

RESPONSE_PROMPT_1 = """
Given Context: {context_data}

Give the best full answer amongest the option to question {question}.
"""

RESPONSE_PROMPT = """
Given Context: {context_data}

Give the best full answer amongest the option to question {question}.

IMPORTANT! JUST OUTPUT THE ANSWER BELOW!

Answer: 
"""

def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_string_by_tiktoken(tokens: list[int], model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    string = ENCODER.decode(tokens)
    return string

def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()