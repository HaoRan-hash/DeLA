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
        self.tao = 1.0
        self.register_buffer('memory', torch.zeros((num_class, length, store_channels)))
        self.attention = SemanticAwareAttention_Mask(query_channels, store_channels, store_channels)
        self.cur_occupy = [0] * num_class
        
        self.dropout = nn.Dropout(0.1)
        self.attn_mlp = nn.Sequential(nn.Conv1d(query_channels, query_channels*4, 1, bias=False),
                                 nn.BatchNorm1d(query_channels * 4, momentum=0.1),
                                 nn.GELU(),
                                 nn.Dropout(0.1),
                                 nn.Conv1d(query_channels*4, query_channels, 1))
        
        self.proj_mlp = nn.Sequential(nn.Conv1d(query_channels, store_channels, 1, bias=False),
                                      nn.BatchNorm1d(store_channels, momentum=0.1),
                                      nn.GELU(),
                                      nn.Conv1d(store_channels, store_channels, 1, bias=False),
                                      nn.BatchNorm1d(store_channels, momentum=0.1),
                                      nn.GELU(),
                                      nn.Conv1d(store_channels, store_channels, 1))
    
    def is_full(self):
        res = True
        for i in range(self.num_class):
            res = (res and (self.cur_occupy[i] == self.length))
        return res
    
    @torch.no_grad()
    def update(self, features, gts, coarse_pred, epoch_ratio):
        """
        features.shape = (b, n, store_channels)
        gts.shape = (b, n)
        coarse_pred.shape = (b, num_class, n)
        """
        coarse_pred = coarse_pred.detach().transpose(1, 2).softmax(dim=-1)
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
    
    def forward(self, features, coarse_pred, mask, gts=None):
        """
        features.shape = (b, n, query_channels)
        coarse_pred.shape = (b, num_class, n)
        mask.shape = (b, 1, num_class)
        return res.shape = (b, n, query_channels)
        """
        b = features.shape[0]
        contrast_loss = 0
        if self.training and (not self.is_full()):
            print('not full')
            return features, contrast_loss
        
        memory_features = self.memory.mean(dim=1).unsqueeze(dim=0).expand(b, -1, -1)
        memory_features = F.normalize(memory_features, dim=-1)
        if gts is not None:
            proj_f = F.normalize(self.proj_mlp(features.transpose(1, 2)), dim=1)   # (b, store_channels, n)
            contrast_map = torch.matmul(memory_features, proj_f) / self.tao
            contrast_loss = F.cross_entropy(contrast_map, gts, ignore_index=255)
        
        reve_features = self.attention(F.normalize(features, dim=-1), memory_features, memory_features, mask, coarse_pred)

        res = features + self.dropout(reve_features)
        res = res + self.attn_mlp(res.transpose(1, 2)).transpose(1, 2)
        
        return res, contrast_loss


class SemanticAwareAttention_Mask(nn.Module):
    def __init__(self, embed_dim, kdim, vdim):
        super(SemanticAwareAttention_Mask, self).__init__()
        self.embed_dim = embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(vdim, embed_dim)
    
    def forward(self, q, k, v, mask, coarse_pred=None, need_weights=False):
        """
        q.shape = (b, n, embed_dim)
        k.shape = (b, num_class, kdim)
        v.shape = (b, num_class, vdim)
        mask.shape = (b, 1, num_class)  float32
        coarse_pred.shape = (b, num_class, n)
        """
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        attn_map = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_map = attn_map + mask
        attn_map = attn_map.softmax(dim=-1)

        if coarse_pred is not None:
            coarse_pred = coarse_pred.detach()
            coarse_pred = coarse_pred.transpose(1, 2).softmax(dim=-1)
            attn_map = torch.softmax(attn_map * coarse_pred + mask, dim=-1)
        
        output = torch.bmm(attn_map, v)
        
        if need_weights:
            return output, attn_map
        else:
            return output
