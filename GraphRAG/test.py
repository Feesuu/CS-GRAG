import networkx as nx
import matplotlib.pyplot as plt
from graspologic.partition import hierarchical_leiden

# 创建一个示例图
G = nx.karate_club_graph()

# 使用 hierarchical_leiden 进行社区检测
partition = hierarchical_leiden(G)

# 打印社区分区结果
print(partition)

import pdb
pdb.set_trace()