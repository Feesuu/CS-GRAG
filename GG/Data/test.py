import networkx as nx
import torch
from torch_geometric.utils import from_networkx

# 创建一个示例的 networkx 图
G = nx.karate_club_graph()

# 假设你有节点特征矩阵 x
x = torch.randn((G.number_of_nodes(), 1433))

# 将 networkx 图转换为 torch_geometric 的 Data 对象
data = from_networkx(G)
import pdb
pdb.set_trace()
data.x = x

# 打印转换后的 Data 对象
print(data)