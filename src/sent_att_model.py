"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import MutilHeadAttention

class SentAttNet(nn.Module):
    def __init__(self, head_num=1, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.attention = MutilHeadAttention(head_num, sent_hidden_size)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    def forward(self, input):
        '''
        forward:
        params:
            :param input: [batch_size, max_sent_length, 2 * word_hidden_size]
        return:
            :return output:
        '''
        f_output, h_output = self.gru(input)

        # 实现Attention机制
        # f_output: [batch_size, max_sent_length, 2*sent_hidden_size]
        att_output, weight = self.attention(f_output, f_output, f_output)

        # output
        output = self.fc(att_output)

        # output: [batch_size, num_classes]
        return output


if __name__ == "__main__":
    abc = SentAttNet()
