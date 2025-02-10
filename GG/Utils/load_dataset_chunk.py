import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json
from Utils.utils import (
    coo_matrix_to_nx_graph,
    get_normalized_laplacian,
    get_one_hot,
    get_rows_or,
    get_row_normalize,
    calculate_query_distance,
    mdhash_id
)

class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        dataset = torch.load(os.path.join(dataset_path, "pyg.pt"))
        jsonl_path = os.path.join(dataset_path, "query_gt.jsonl")
        with open(jsonl_path, 'r') as f:
            query_gt = list(map(lambda x: json.loads(x), f.readlines()))

        self.adjs = dataset[0]
        self.x = dataset[1]
        self.chunk_content = dataset[2]
        self.node_nums = len(self.x)
        # "query": structure_query_embedding,
        # "attribute_embedding": attribute_query_embedding, 
        # "ground_truth": ground_truth_score
        
        self.query_attribute = list(map(
            lambda x: torch.tensor(x["attribute_embedding"]).unsqueeze(1).to(torch.float32), query_gt
        ))
        
        self.labels = list(
            map(
                lambda x: torch.tensor(x["ground_truth"]).unsqueeze(1).to(torch.float32), query_gt
            )
        )
        
        self.graph = coo_matrix_to_nx_graph(self.adjs)
        self.normalized_laplacian_adj = get_normalized_laplacian(self.adjs)
        self.row_normalized_features = get_row_normalize(self.x)
        
        query = list(map(lambda x: get_one_hot(list(set(x['query'])), self.node_nums), query_gt)) # list[N*1]
        query = torch.cat(list(map(lambda x: x.transpose(0, 1), query)), dim=0) # query_num * N
        query_distance = calculate_query_distance(graph=self.graph, 
                                                  cur_in=query, # query_num * N
                                                  exist_dis={}) # query_num * N
        self.query = list(
            map(
                lambda x: x, 
                torch.stack((query, query_distance), dim=-1)
            )
        )
        
        self.question = list(
            map(
                lambda x: x["question"], query_gt
            )
        )
        # import pdb
        # pdb.set_trace()
        # self.test()

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        return self.query[idx], self.query_attribute[idx], self.labels[idx], self.question[idx]
    
    @property
    def adj_n(self):
        return self.normalized_laplacian_adj.to(torch.float32)
    
    @property
    def x_n(self):
        return self.row_normalized_features.to(torch.float32)
    
    @property
    def adj(self):
        return self.adjs.to(torch.float32)
    
    @property
    def chunk(self):
        return self.chunk_content


if __name__ == "__main__":
    dataset_path = '../Data/dataset/chunk/'
    dataset = CustomDataset(dataset_path)
    
    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # for batch in train_loader:
    #     samples, labels = batch

    #     # import pdb
    #     # pdb.set_trace()