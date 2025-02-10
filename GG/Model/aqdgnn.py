import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #support = input.matmul(self.weight)
        # import pdb
        # pdb.set_trace()
        support = torch.matmul(input, self.weight)
        if input.dim() == 2:
            #output = torch.sparse.mm(adj, support)
            output = torch.spmm(adj, support)
            output = output + self.bias if self.bias is not None else output
        else:
            # batch size
            output = torch.stack(
                list(
                    map(
                        lambda x: torch.sparse.mm(adj, x)  + self.bias if self.bias is not None else torch.sparse.mm(adj, x), 
                        support
                    )
                )
            )
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfLoop(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(SelfLoop, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #support = input.matmul(self.weight)
        support = torch.matmul(input, self.weight)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AQD_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AQD_GCN, self).__init__()

        ############################
        self.graph_encoder1 = GraphConvolution(nfeat, nhid)
        self.structure_encoder1 = GraphConvolution(2, nhid)
        self.attribute_encoder1 = GraphConvolution(1, nhid)
        self.self_GE1 = SelfLoop(nfeat, nhid)
        self.self_SE1 = SelfLoop(2, nhid)
        self.self_AE1 = SelfLoop(1, nhid)

        self.bn_GE1 = nn.BatchNorm1d(nhid)
        self.bn_SE1 = nn.BatchNorm1d(nhid)
        self.bn_AE1 = nn.BatchNorm1d(nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.cnd1 = SelfLoop(3 * nhid, nhid)

        ############################

        self.graph_encoder2 = GraphConvolution(nhid, nhid)
        self.structure_encoder2 = GraphConvolution(nhid, nhid)
        self.attribute_encoder2 = GraphConvolution(nhid, nhid)
        self.self_GE2 = SelfLoop(nhid, nhid)
        self.self_SE2 = SelfLoop(nhid, nhid)
        self.self_AE2 = SelfLoop(nhid, nhid)

        self.bn_GE2 = nn.BatchNorm1d(nhid)
        self.bn_SE2 = nn.BatchNorm1d(nhid)
        self.bn_AE2 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.cnd2 = SelfLoop(3 * nhid, nhid)

        ############################

        self.graph_encoder3 = GraphConvolution(nhid, nclass)
        self.structure_encoder3 = GraphConvolution(nhid, nclass)
        self.attribute_encoder3 = GraphConvolution(nhid, nclass)
        self.self_GE3 = SelfLoop(nhid, nclass)
        self.self_SE3 = SelfLoop(nhid, nclass)
        self.self_AE3 = SelfLoop(nhid, nclass)

        self.cnd3 = SelfLoop(3 * nclass, nclass)

        ############################
        self.dropout = dropout

    def forward(self, node_input, att_input, adj, Fadj, feat, training=True):
        # import pdb
        # pdb.set_trace()
        model1 = self.graph_encoder1(feat, adj) + self.self_GE1(feat) # hG1
        model2 = self.structure_encoder1(node_input, adj) + self.self_SE1(node_input) # hQ1
        model3 = self.attribute_encoder1(att_input, Fadj) # hN1

        ######### expand ##########
        batchsize = model2.shape[0]
        model1 = model1.unsqueeze(0).expand(batchsize, -1, -1)
        
        model = torch.cat([model1, model2], 2)
        model = torch.cat([model, model3], 2)
        model = self.cnd1(model) # hFF1


        #model = self.bn1(model)
        model = self.bn1(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ############################

        #model1 = self.bn_GE1(model1)
        model1=self.bn_GE1(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1 = self.graph_encoder2(model1, adj) + self.self_GE2(model1) #hG2
        model2 = self.structure_encoder2(model, adj) + self.self_SE2(model) # hQ2

        #model3 = Fadj.transpose(1, 0).matmul(model) + self.self_AE1(att_input) # hA1
        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE1(att_input)
        #model_AE = self.bn_AE1(model3)
        model_AE = self.bn_AE1(model3.transpose(-1, -2)).transpose(-1, -2)
        model_AE = F.relu(model_AE)
        model_AE = F.dropout(model_AE, self.dropout, training=training)
        model3 = self.attribute_encoder2(model_AE, Fadj) # hN2

        model = torch.cat([model1, model2], 2)
        model = torch.cat([model, model3], 2)
        model = self.cnd2(model) # hFF2

        #model = self.bn2(model)
        model = self.bn2(model.transpose(-1, -2)).transpose(-1, -2)
        model = F.relu(model)
        model = F.dropout(model, self.dropout, training=training)

        ################################################################################################################

        #model1 = self.bn_GE2(model1)
        model1=self.bn_GE2(model1.transpose(-1, -2)).transpose(-1, -2)
        model1 = F.relu(model1)
        model1 = F.dropout(model1, self.dropout, training=training)
        model1 = self.graph_encoder3(model1, adj) + self.self_GE3(model1) # hG3
        model2 = self.structure_encoder3(model, adj) + self.self_SE3(model) # hQ3
        
        # model3 = Fadj.transpose(1,0).matmul(model) + self.self_AE2(model_AE) # hA2
        # model3 = self.bn_AE2(model3)
        model3 = Fadj.transpose(-1, -2).matmul(model) + self.self_AE2(model_AE)
        model3 = self.bn_AE2(model3.transpose(-1, -2)).transpose(-1, -2)

        model3 = F.relu(model3)
        model3 = F.dropout(model3, self.dropout, training=training)
        model3 = self.attribute_encoder3(model3, Fadj) # hN3

        model = torch.cat([model1, model2], 2)
        model = torch.cat([model, model3], 2)
        model = self.cnd3(model)  # hFF3

        return torch.sigmoid(model)



