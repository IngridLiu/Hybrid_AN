"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.att_model import *

# use ori attention to build model
class Ori_HAN(nn.Module):
    def __init__(self,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Ori_HAN, self).__init__()
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.days_num = days_num
        self.num_classes = num_classes
        self.pretrained_word2vec_path = pretrained_word2vec_path
        self.dropout = dropout
        self.days_att_net = Ori_DaysNewsAttNet(days_hidden_size=days_hidden_size,
                                               news_hidden_size=news_hidden_size,
                                               dropout=dropout)
        self.news_att_net = Ori_NewsAttNet(pretrained_word2vec_path, news_hidden_size)

    def forward(self, days_newses_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        newses_output_list = []
        days_newses_input = days_newses_input.permute(1, 0, 2, 3, 4)
        for newses_input in days_newses_input:
            # newses_input: [batch_size, max_news_length, max_sent_length, max_word_length]
            newses_output = self.news_att_net(newses_input)
            newses_output = newses_output.unsqueeze(0)
            newses_output_list.append(newses_output)
        days_newses_output = torch.cat(newses_output_list, 0).permute(1, 0, 2)

        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_outputs = self.days_att_net(days_newses_output)

        # sent_output:
        return days_newses_outputs

class Sent_Ori_HAN(nn.Module):
    def __init__(self,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 sent_hidden_size,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Sent_Ori_HAN, self).__init__()
        self.days_num = days_num
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_classes = num_classes
        self.pretrained_word2vec_path = pretrained_word2vec_path
        self.dropout = dropout
        self.days_att_net = Ori_DaysNewsAttNet(days_hidden_size=days_hidden_size, news_hidden_size=news_hidden_size)
        self.news_att_net = Sent_Ori_NewsAttNet(news_hidden_size=news_hidden_size, sent_hidden_size=sent_hidden_size)
        self.sent_att_net = Ori_SentAttNet(word2vec_path=pretrained_word2vec_path, sent_hidden_size=4)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes)

    def forward(self, days_newses_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        days_newses_output_list = []
        days_newses_input = days_newses_input.permute(1, 0, 2, 3, 4)
        for newses_input in days_newses_input:
            # i: [batch_size, max_news_length, max_sent_length, max_word_length]
            news_output_list = []
            newses_input = newses_input.permute(1, 0, 2, 3)
            for news_input in newses_input:
                # news_input: [batch_size, max_news_length, max_sent_length, max_word_length]
                # sent attention net
                news_output = self.sent_att_net(news_input)
                news_output = news_output.unsqueeze(0)
                news_output_list.append(news_output)
            newses_output = torch.cat(news_output_list, 0)
            newses_output = newses_output.permute(1, 0, 2)
            # news attention net
            newses_output = self.news_att_net(newses_output)
            newses_output = newses_output.unsqueeze(0)
            days_newses_output_list.append(newses_output)
        days_newses_outputs = torch.cat(days_newses_output_list, 0).permute(1, 0, 2)

        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_outputs = self.days_att_net(days_newses_outputs)
        days_newses_outputs = self.fc(days_newses_outputs)

        # sent_output:
        return days_newses_outputs


# muilty head han
class Muil_HAN(nn.Module):
    def __init__(self,head_num,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Muil_HAN, self).__init__()
        self.head_num = head_num
        self.days_num = days_num
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.num_classes = num_classes
        self.pretrained_word2vec_path = pretrained_word2vec_path
        self.dropout = dropout
        self.days_att_net = Muil_DaysNewsAttNet(head_num = head_num,
                                                days_hidden_size=days_hidden_size,
                                                news_hidden_size=news_hidden_size,
                                                dropout=dropout)
        self.news_att_net = Muil_NewsAttNet(word2vec_path=pretrained_word2vec_path,
                                            head_num=head_num,
                                            news_hidden_size=news_hidden_size,
                                            dropout=dropout)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes)



    def forward(self, days_newses_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        newses_output_list = []
        days_newses_input = days_newses_input.permute(1, 0, 2, 3, 4)
        for newses_input in days_newses_input:
            # newses_input: [batch_size, max_news_length, max_sent_length, max_word_length]
            newses_output = self.news_att_net(newses_input)
            newses_output = newses_output.unsqueeze(0)
            newses_output_list.append(newses_output)
        days_newses_output = torch.cat(newses_output_list, 0).permute(1, 0, 2)
        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_outputs = self.days_att_net(days_newses_output)
        days_newses_outputs = self.fc(days_newses_outputs)

        # sent_output:
        return days_newses_outputs

#
class Sent_Muil_HAN(nn.Module):
    def __init__(self,head_num,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 sent_hidden_size,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Sent_Muil_HAN, self).__init__()
        self.head_num = head_num
        self.days_num = days_num
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_classes = num_classes
        self.pretrained_word2vec_path = pretrained_word2vec_path
        self.dropout = dropout
        self.days_att_net = Muil_DaysNewsAttNet(head_num=head_num, days_hidden_size=days_hidden_size, news_hidden_size=news_hidden_size)
        self.news_att_net = Sent_Muil_NewsAttNet(head_num = head_num, news_hidden_size=news_hidden_size, sent_hidden_size=sent_hidden_size)
        self.sent_att_net = Muil_SentAttNet(word2vec_path=pretrained_word2vec_path, head_num = head_num, sent_hidden_size=sent_hidden_size)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes)

    def forward(self, days_newses_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        days_newses_output_list = []
        days_newses_input = days_newses_input.permute(1, 0, 2, 3, 4)
        for newses_input in days_newses_input:
            # newses_input: [batch_size, max_news_length, max_sent_length, max_word_length]
            news_output_list = []
            newses_input = newses_input.permute(1, 0, 2, 3)
            for news_input in newses_input:
                # news_input: [batch_size, max_news_length, max_sent_length, max_word_length]
                # sent attention net
                news_output = self.sent_att_net(news_input)
                news_output = news_output.unsqueeze(0)
                news_output_list.append(news_output)
            newses_output = torch.cat(news_output_list, 0).permute(1, 0, 2)
            # news attention net
            newses_output = self.news_att_net(newses_output)
            newses_output = newses_output.unsqueeze(0)
            days_newses_output_list.append(newses_output)
        days_newses_outputs = torch.cat(days_newses_output_list, 0).permute(1, 0, 2)

        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_outputs = self.days_att_net(days_newses_outputs)
        days_newses_outputs = self.fc(days_newses_outputs)

        # sent_output:
        return days_newses_outputs