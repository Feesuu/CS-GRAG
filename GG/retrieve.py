from Utils.load_dataset_chunk import CustomDataset
from Model.aqdgnn import AQD_GCN
import torch
import torch.nn as nn
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Utils.utils import calculate_metrics, calculate_metrics_chunk, SYSTEM_PROMPT, RESPONSE_PROMPT, encode_string_by_tiktoken, decode_string_by_tiktoken, mdhash_id
import torch.nn.functional as F
import random
import numpy as np
import os
import json

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

# set_seed(11)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(max_token_len=6000):
    parser = argparse.ArgumentParser(description='GNNRAG Training Script')
    parser.add_argument('--dataset_path', type=str,default="./Data/dataset/chunk/", help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,help='weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # 解析参数
    args = parser.parse_args()

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 检查CUDA是否可用
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    logger.info(f"Using device: {device}")

    # 数据集划分
    dataset = CustomDataset(args.dataset_path)
    #train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=args.seed)
    
    #test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 数据
    adj_n = dataset.adj_n.to(device)
    x_n = dataset.x_n.to(device)
    chunks = dataset.chunk
    #adj = dataset.adj
    
    # model
    model = AQD_GCN(nfeat=x_n.shape[1], 
                    nhid=args.hidden_size, 
                    nclass=1, 
                    dropout=0.1)
    model_path = '/home/yaodong/codes/GNNRAG/GG/Data/dataset/chunk/model.pth'
    model.load_state_dict(torch.load(model_path))
    logging.info("Model loaded from {model_path}")
    
    model = model.to(device)

    ############# Testing #############
    with torch.no_grad():
        total_mse = 0
        total_mae = 0
        total_r2s = 0
        num_batches = 0
        
        jsonl_result = []
        for query, query_attribute, labels, question in test_loader:
            query, query_attribute, labels = query.to(device), query_attribute.to(device), labels.to(device)
            question = question[0]
            outputs = model(node_input=query, 
                            att_input=query_attribute, 
                            adj=adj_n, 
                            Fadj=x_n, 
                            feat=x_n,
                            training=False)
            # 计算指标
            mse, mae, r2 = calculate_metrics_chunk(y_pred=outputs, y_true=labels)
            # 记录
            total_mse += mse
            total_mae += mae
            total_r2s += r2
            num_batches += 1
            
            # import pdb
            # pdb.set_trace()
            
            md5_id = mdhash_id(question)
            
            indices = torch.topk(outputs.squeeze(0).squeeze(1), 10, largest=True).indices
            retrieved_context = list(map(lambda x: chunks[x]['content'], indices))
            retrieved_context = "\n\n".join(retrieved_context)
            
            question_token_len = len(encode_string_by_tiktoken(question))
            left_token_len = max_token_len - 103 - 70 - question_token_len
            user_prompt_token_len = len(encode_string_by_tiktoken(retrieved_context))
            
            if user_prompt_token_len >= left_token_len:
                left_prompt = decode_string_by_tiktoken(encode_string_by_tiktoken(retrieved_context)[:left_token_len])
            else:
                left_prompt = retrieved_context
            
            res = RESPONSE_PROMPT.format(context_data=left_prompt, question=question)
                
            result = {
                "question_id": md5_id,
                "md5_id": md5_id,
                "prompt": SYSTEM_PROMPT.format(res),
                "code": "",
            }
            
            jsonl_result.append(result)
        

        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        avg_r2s = total_r2s / num_batches
        logger.info(f"MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R2-score: {avg_r2s:.4f}")
        
        with open('/home/yaodong/codes/GNNRAG/GG/Data/dataset/chunk/generation.jsonl', "w") as file:
            for item in jsonl_result:
                json.dump(item, file)
                file.write("\n")

if __name__ == '__main__':
    # 设置随机种子
    set_seed(42)
    main()