"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import matrix_mul, element_wise_mul
from src.attention import MutilHeadAttention

import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, head_num=1, hidden_size=8):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.emb = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self.attention = MutilHeadAttention(head_num, hidden_size)

    def forward(self, input):
        output = self.emb(input)
        f_output, h_output = self.gru(output.float())  # feature output and hidden state output

        # Attention机制
        # f_output: [batch_size, max_word_length, 2*word_hidden_size]
        output, weight = self.attention(f_output, f_output, f_output)

        # output: [batch_size, 2 * word_hidden_size]
        return output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
