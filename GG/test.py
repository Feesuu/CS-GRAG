
from Model.aqdgnn import AQD_GCN
import torch
model = AQD_GCN(nfeat=1024, 
                nhid=256, 
                nclass=1, 
                dropout=0.1)

# save the model
# 保存模型
model_path = '/home/yaodong/codes/GNNRAG/GG/Data/dataset/chunk/model.pth'
model.load_state_dict(torch.load(model_path))
print(f"Model loaded from {model_path}")

print(model)