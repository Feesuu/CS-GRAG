import torch
import json
data_list = torch.load("./dataset/cora/pyg.pt")

labels = data_list[2]
torch.manual_seed(0)

print(labels, torch.min(labels), torch.max(labels))
num_class = torch.max(labels)-torch.min(labels)+1
print(num_class)
communities = [[i for i in range(labels.shape[0]) if labels[i]==j] for j in range(num_class)]

selected_queries = []
ground_truth = []

query_nums = 300

for i in range(query_nums):
    num_node = torch.randint(1, 3, (1,)).item()
    selected_class = torch.randint(0, num_class, (1,)).item()
    selected_nodes = []
    for j in range(num_node):
        selected_node = torch.randint(0, len(communities[selected_class]), (1,)).item()
        selected_nodes.append(communities[selected_class][selected_node])
    selected_queries.append(selected_nodes)
    ground_truth.append(communities[selected_class])

# import pdb
# pdb.set_trace()

with open("./dataset/cora/query_gt.jsonl", "w") as f:
    for i in range(query_nums):
        f.write(
            json.dumps(
                {
                    "query": selected_queries[i],
                    "ground_truth": ground_truth[i]
                }
            ) + "\n"
        )

# query_file=open("cora.query","w")
# gt_file = open("cora.gt", "w")

# for i in range(len(selected_queries)):
#     for j in range(len(selected_queries[i])):
#         query_file.write(str(selected_queries[i][j]))
#         query_file.write(" ")
#     query_file.write("\n")
#     for j  in range(len(ground_truth[i])):
#         gt_file.write(str(ground_truth[i][j]))
#         gt_file.write(" ")
#     gt_file.write("\n")

# adj = data_list[0]
# # print(adj)
# coalesced_tensor = adj.coalesce()
# index = coalesced_tensor.indices()
# # print(index)

# edge_file = open("cora.edges", "w")
# for i in range(index.shape[1]):
#     edge_file.write(str(index[0][i].item()))
#     edge_file.write(" ")
#     edge_file.write(str(index[1][i].item()))
#     edge_file.write("\n")





