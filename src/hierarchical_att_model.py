"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, head_num, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.head_num = head_num
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, head_num, word_hidden_size)
        self.sent_att_net = SentAttNet(head_num, sent_hidden_size, word_hidden_size, num_classes)

    def forward(self, input):
        '''
        params:
            :param input: [batch_size, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        word_output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            # i: [batch_size, max_word_length]
            word_output = self.word_att_net(i)
            word_output = word_output.unsqueeze(0)
            word_output_list.append(word_output)
        word_outputs = torch.cat(word_output_list, 0).permute(1, 0, 2)

        # word_outputs: [batch_size, max_sent_length, 2 * hidden_size]
        sent_output = self.sent_att_net(word_outputs)

        # sent_output:
        return sent_output
