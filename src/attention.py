import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math


# scaled attention
class Attention():
    # basic attention
    def basic_attention(self, query, key, value, mask=None, dropout=None):
        '''
        Compute 'Scaled Dot Product Attention

        params:
            :param query: [batch_size, max_length, 2*hidden_size]
            :param key: [batch_size, max_length, 2*hidden_size]
            :param value: [batch_size, max_length, 2*hidden_size]
            :param mask:
            :param dropout:
        return:
            :return output: [batch_size, max_length, 2*hidden_size]
            :return p_attn: []
        '''

        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value).sum(dim=1)

        # output: [batch_size, 2*hidden_size]
        # p_attn: [batch_size, max_length, max_length]
        return output, p_attn

    # scaled attention
    def scaled_attention(self, query, key, value, mask=None, dropout=None):
        '''
        Compute 'Scaled Dot Product Attention

        params:
            :param query: [batch_size, max_length, 2*hidden_size]
            :param key: [batch_size, max_length, 2*hidden_size]
            :param value: [batch_size, max_length, 2*hidden_size]
            :param mask:
            :param dropout:
        return:
            :return output: [batch_size, max_length, 2*hidden_size]
            :return p_attn: []
        '''

        hidden_size = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value).sum(dim=1)

        # output: [batch_size, 2*hidden_size]
        # p_attn: [batch_size, max_length, max_length]
        return output, p_attn


# basic attention
class BasicAttention(nn.Module, Attention):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attention = Attention()
        self.linear = nn.Linear(2*hidden_size, 2*hidden_size)

    def forward(self, query, key, value):
        '''
        froword:

        params:
            :param query: [batch_size, max_lenght, 2*word_hidden_size]
            :param key: [batch_size, max_lenght, 2*word_hidden_size]
            :param value: [batch_size, max_lenght, 2*word_hidden_size]
        return:
            :return outputs: []
            :return weights: []
        '''

        outputs, weights = self.attention.basic_attention(query, key, value)
        outputs = self.linear(outputs)

        # outputs: [batch_size, 2*hidden_size]
        # weights: [batch_size, max_length, max_word_length]
        return outputs, weights



# mutil-head attention
class MutilHeadAttention(nn.Module, Attention):
    def __init__(self, head_num, hidden_size, dropout=0.1):
        "初始化时指定头数h和模型维度self.d_model"
        super(MutilHeadAttention, self).__init__()
        # 按照中文的简化,我们让d_v与hidden_size相等
        self.hidden_size = hidden_size
        self.d_model = head_num * hidden_size
        self.head_num = head_num
        self.linear = nn.Linear(2 * hidden_size, self.d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = Attention()
        self.end_linear = nn.Linear(self.d_model, 2 * hidden_size)

    def forward(self, query, key, value, mask=None):
        "实现多头注意力模型"
        # 第一步是计算一下mask
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        # 第二步是将这一批次的数据进行变形 self.d_model => h x hidden_size
        query = self.linear(query).view(batch_size, -1, self.head_num, self.hidden_size) # [batch_size, max_length, d_model]
        key = self.linear(key).view(batch_size, -1, self.head_num, self.hidden_size)
        value = self.linear(value).view(batch_size, -1, self.head_num, self.hidden_size)
        # 第三步，针对所有变量计算scaled dot product attention
        x, self.attn = self.attention.scaled_attention(query, key, value, mask=mask, dropout=self.dropout)
        # 最后，将attention计算结果串联在一起，其实对张量进行一次变形：
        x = x.transpose(1, 2).contiguous().view(batch_size, self.head_num * self.hidden_size)
        output = self.end_linear(x)
        return output, self.attn


