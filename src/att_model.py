"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn

from src.attention import BasicAttention, MutilHeadAttention

import pandas as pd
import numpy as np
import csv


# Ori_DaysAttNet
class Ori_DaysAttNet(nn.Module):
    def __init__(self, days_hidden_size=16, news_hidden_size=8, num_classes=2):
        super(Ori_DaysAttNet, self).__init__()
        self.gru = nn.GRU(2 * news_hidden_size, days_hidden_size, bidirectional=True)
        self.attention = BasicAttention(days_hidden_size)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, days_num, 2 * news_hidden_size]
        return:
            :return output:
        '''
        f_output, h_output = self.gru(input)
        # 实现Attention机制
        # f_output: [batch_size, days_num, 2 * days_hidden_size]
        att_output, weight = self.attention(f_output, f_output, f_output)
        # output
        output = self.fc(att_output)

        # output: [batch_size, num_classes]
        return output

# DaysAttNet
class Muil_DaysAttNet(nn.Module):
    def __init__(self, head_num=1, days_hidden_size=16, news_hidden_size=8, num_classes=2, dropout=0.1):
        super(Muil_DaysAttNet, self).__init__()
        self.dropout = dropout
        self.head_num = head_num
        self.gru = nn.GRU(2 * news_hidden_size, days_hidden_size, bidirectional=True)
        self.attention = MutilHeadAttention(head_num, 2 * days_hidden_size, dropout)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, days_num, 2 * news_hidden_size]
        return:
            :return output:
        '''
        f_output, h_output = self.gru(input)
        # 实现Attention机制
        # f_output: [batch_size, days_num, 2*days_hidden_size]
        att_output, weight = self.attention(f_output, f_output, f_output)
        # output
        output = self.fc(att_output)

        # output: [batch_size, num_classes]
        return output

# News Attention Layer
# Ori_NewsAttNet
class Ori_NewsAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=8):
        super(Ori_NewsAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.emb = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.linear = nn.Linear(embed_size, 2* hidden_size)
        self.attention = BasicAttention(hidden_size)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, max_news_length, max_sent_length, max_word_length]
        return:
            :return output:
        '''
        emb_output = self.emb(input)    # [batch_size, max_news_length, max_sent_length, max_word_length, embedding_size]
        sent_avg = emb_output.mean(dim=-2)  # [batch_size, max_news_length, max_sent_length, embedding_size]
        news_avg = sent_avg.mean(dim=-2).float()    # [batch_size, max_news_length, embedding_size]
        news_output = self.linear(news_avg) # [batch_size, max_news_length, 2 * news_hidden_size]

        # Attention机制
        output, weight = self.attention(news_output, news_output, news_output)

        # output:[batch_size, 2 * news_hidden_size]
        return output

# News Attention Net with muilty head
class Muil_NewsAttNet(nn.Module):
    def __init__(self, word2vec_path, head_num=1, hidden_size=8):
        super(Muil_NewsAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
        self.head_num = head_num
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.linear = nn.Linear(embed_size, 2 * hidden_size)
        self.attention = MutilHeadAttention(head_num=head_num, hidden_size=hidden_size, dropout=0)

    def forward(self, input):
        emb_output = self.emb(input)
        sent_avg = emb_output.avg(dim=-2)
        news_avg = sent_avg.avg(dim=-2).float()
        news_output = self.linear(news_avg)

        # Attention机制
        output, weight = self.attention(news_output, news_output, news_output)

        return output

# News Attention Net with sentense layer and basic attention
class Sent_Ori_NewsAttNet(nn.Module):
    def __init__(self, news_hidden_size=8, sent_hidden_size=4):
        super(Sent_Ori_NewsAttNet, self).__init__()
        self.news_hidden_size = news_hidden_size
        self.sent_hidden_size = sent_hidden_size

        self.gru = nn.GRU(2 * sent_hidden_size, news_hidden_size, bidirectional=True)
        self.attention = BasicAttention(news_hidden_size)
        self.fc = nn.Linear(2 * news_hidden_size, 2 * news_hidden_size)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, days_num, 2 * news_hidden_size]
        return:
            :return output:
        '''
        f_output, h_output = self.gru(input)
        # 实现Attention机制
        # f_output: [batch_size, days_num, 2*days_hidden_size]
        att_output, weight = self.attention(f_output, f_output, f_output)
        # output
        output = self.fc(att_output)

        # output: [batch_size, num_classes]
        return output

# News Attention Net with sentense layer and basic attention
class Sent_Muil_NewsAttNet(nn.Module):
    def __init__(self, head_num = 1, news_hidden_size=8, sent_hidden_size=4):
        super(Ori_DaysAttNet, self).__init__()
        self.head_num = head_num
        self.news_hidden_size = 8
        self.sent_hidden_size = 4

        self.gru = nn.GRU(2 * sent_hidden_size, news_hidden_size, bidirectional=True)
        self.attention = MutilHeadAttention(head_num=head_num, news_hidden_size = news_hidden_size)
        self.fc = nn.Linear(2 * news_hidden_size, 2 * news_hidden_size)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, days_num, 2 * news_hidden_size]
        return:
            :return output:
        '''
        f_output, h_output = self.gru(input)
        # 实现Attention机制
        # f_output: [batch_size, days_num, 2*days_hidden_size]
        att_output, weight = self.attention(f_output, f_output, f_output)
        # output
        output = self.fc(att_output)

        # output: [batch_size, num_classes]
        return output

# Sent Attention Layer with basic attention
class Ori_SentAttNet(nn.Module):
    def __init__(self, word2vec_path, sent_hidden_size=4):
        super(Ori_SentAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.emb = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.linear = nn.Linear(embed_size, 2* sent_hidden_size)
        self.attention = BasicAttention(sent_hidden_size)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, days_num, 2 * news_hidden_size]
        return:
            :return output:
        '''
        emb_output = self.emb(input)
        sent_avg = emb_output.mean(dim=-2).float()  # [batch_size, max_sent_length, embedding_size]
        sent_output = self.linear(sent_avg)

        # Attention机制
        output, weight = self.attention(sent_output, sent_output, sent_output)

        return output

# Sent Attention Layer with muilty head attention
class Muil_SentAttNet(nn.Module):
    def __init__(self, word2vec_path, head_num=1, sent_hidden_size=4):
        super(Muil_NewsAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
        self.head_num = head_num
        self.hidden_size = sent_hidden_size

        self.emb = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.linear = nn.Linear(embed_size, 2 * sent_hidden_size)
        self.attention = MutilHeadAttention(head_num=head_num, hidden_size=sent_hidden_size, dropout=0)

    def forward(self, input):
        emb_output = self.emb(input)
        sent_avg = emb_output.avg(dim=-2)
        sent_output = self.linear(sent_avg).float()

        # Attention机制
        output, weight = self.attention(sent_output, sent_output, sent_output)

        return output