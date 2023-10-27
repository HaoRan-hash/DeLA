import torch
from torch import nn
import torch.nn.functional as F
import math


class Memory(nn.Module):
    def __init__(self, num_class, length, store_channels, query_channels, base_miu=0.1, end_miu=0.001):
        super(Memory, self).__init__()
        self.num_class = num_class
        self.length = length
        self.store_channels = store_channels
        self.query_channles = query_channels
        self.base_miu = base_miu
        self.end_miu = end_miu
        self.poly_power = 0.9
        self.tao = 0.1
        self.register_buffer('memory', torch.zeros((num_class, length, store_channels)))
        self.attention = SemanticAwareAttention(query_channels, store_channels, store_channels)
        self.cur_occupy = [0] * num_class
        
        self.dropout = nn.Dropout(0.1)
        self.attn_mlp = nn.Sequential(nn.Linear(query_channels, query_channels * 4, bias=False),
                                      nn.BatchNorm1d(query_channels * 4, momentum=0.02),
                                      nn.GELU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(query_channels * 4, query_channels))
        
        self.proj_mlp = nn.Sequential(nn.Linear(query_channels, store_channels, bias=False),
                                      nn.BatchNorm1d(store_channels, momentum=0.02),
                                      nn.GELU(),
                                      nn.Linear(store_channels, store_channels, bias=False),
                                      nn.BatchNorm1d(store_channels, momentum=0.02),
                                      nn.GELU(),
                                      nn.Linear(store_channels, store_channels, 1))
    
    def is_full(self):
        res = True
        for i in range(self.num_class):
            res = (res and (self.cur_occupy[i] == self.length))
        return res
    
    @torch.no_grad()
    def update(self, features, gts, coarse_pred, epoch_ratio):
        """
        features.shape = (n, store_channels)
        gts.shape = (n, )
        coarse_pred.shape = (n, num_class)
        """
        coarse_pred = coarse_pred.detach().softmax(dim=-1)
        _, pred_labels = coarse_pred.max(dim=-1)
        
        mask1 = (pred_labels == gts)
        cur_miu = math.pow(1 - epoch_ratio, self.poly_power) * (self.base_miu - self.end_miu) + self.end_miu
        
        for i in range(self.num_class):
            mask2 = (gts == i)
            mask = (mask1 & mask2)
            cur_features = features[mask]
            n = len(cur_features)
            
            # debug
            # with open('seg/pointnext_contrast.log', mode='a') as f:
            #     f.write(f'class {i}, {n} samples\n')
            
            if n != 0 :   # 如果存在该类的feature
                # 模仿dataset的选取策略
                if n >= self.length:
                    choice = torch.arange(0, self.length, 1, dtype=torch.long)
                else:
                    temp = torch.arange(0, n, 1, dtype=torch.long)
                    pad_choice = torch.randint(0, n, (self.length-n, ), dtype=torch.long)
                    choice = torch.cat((temp, pad_choice))
                
                if self.cur_occupy[i] != self.length:   # 该类的memory未满
                    self.memory[i] = cur_features[choice]
                    self.cur_occupy[i] += self.length
                else:
                    self.memory[i] = cur_features[choice] * cur_miu + self.memory[i] * (1 - cur_miu)
    
    def forward(self, features, coarse_pred, gts=None):
        """
        features.shape = (n, query_channels)
        coarse_pred.shape = (n, num_class)
        return res.shape = (n, query_channels)
        """
        contrast_loss = 0
        if self.training and (not self.is_full()):
            print('not full')
            return features, contrast_loss
        
        memory_features = self.memory.mean(dim=1)
        memory_features = F.normalize(memory_features, dim=-1)
        if gts is not None:
            proj_f = F.normalize(self.proj_mlp(features), dim=1)   # (n, store_channels)
            contrast_map = torch.matmul(proj_f, memory_features) / self.tao
            contrast_loss = F.cross_entropy(contrast_map, gts, ignore_index=20)
        
        reve_features = self.attention(F.normalize(features, dim=-1), memory_features, memory_features, coarse_pred)

        res = features + self.dropout(reve_features)
        res = res + self.attn_mlp(res)
        
        return res, contrast_loss


class SemanticAwareAttention(nn.Module):
    def __init__(self, embed_dim, kdim, vdim):
        super(SemanticAwareAttention, self).__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(vdim, embed_dim)
    
    def forward(self, q, k, v, coarse_pred=None, need_weights=False):
        """
        q.shape = (n, embed_dim)
        k.shape = (num_class, kdim)
        v.shape = (num_class, vdim)
        coarse_pred.shape = (n, num_class)
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        attn_map = torch.mm(q, k.T) / math.sqrt(self.embed_dim)
        attn_map = attn_map.softmax(dim=-1)
        
        if coarse_pred is not None:
            coarse_pred = coarse_pred.detach()
            coarse_pred = coarse_pred.softmax(dim=-1)
            attn_map = torch.softmax(attn_map * coarse_pred, dim=-1)
            # attn_map = attn_map * coarse_pred
            # attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)
        
        output = torch.mm(attn_map, v)
        
        if need_weights:
            return output, attn_map
        else:
            return output
