from src.att_model import *


# muilty head han
class Muil_Stock_HAN(nn.Module):
    def __init__(self, head_num,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 stock_hidden_size,
                 stock_length,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Muil_Stock_HAN, self).__init__()
        self.head_num = head_num
        self.days_num = days_num
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.stock_hidden_size = stock_hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.days_news_att_net = Muil_DaysNewsAttNet(days_hidden_size, news_hidden_size, dropout=dropout)
        self.news_att_net = Muil_NewsAttNet(pretrained_word2vec_path, news_hidden_size, dropout=dropout)
        self.days_stock_att_net = Muil_DaysStockAttNet(days_hidden_size, stock_hidden_size, dropout=dropout)
        self.stock_att_net = Muil_StockAttNet(head_num=head_num, stock_length=stock_length, stock_hidden_size=stock_hidden_size, dropout=dropout)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes, dropout=dropout)

    def forward(self, days_newses_input, days_stock_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        days_output_list = []
        # news network
        days_newses_output_list = []
        days_newses_input = days_newses_input.permute(1, 0, 2, 3, 4)
        for newses_input in days_newses_input:
            # newses_input: [batch_size, max_news_length, max_sent_length, max_word_length]
            newses_output = self.news_att_net(newses_input)
            newses_output = newses_output.unsqueeze(0)
            days_newses_output_list.append(newses_output)
        days_newses_output = torch.cat(days_newses_output_list, 0).permute(1, 0, 2)
        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_output = self.days_news_att_net(days_newses_output)
        days_newses_output = days_newses_output.unsqueeze(0)
        days_output_list.append(days_newses_output)

        # stock network
        days_stock_output_list = []
        days_stock_input = days_stock_input.permute(1, 0 ,2)
        for stock in days_stock_input:
            # stock: [batch_size, stock_length]
            stock_output = self.stock_att_net(stock)
            stock_output = stock_output.unsqueeze(0)
            days_stock_output_list.append(stock_output)
        days_stock_output = torch.cat(days_stock_output_list, 0).permute(1, 0, 2)
        days_stock_output = self.days_stock_att_net(days_stock_output)
        days_stock_output = days_stock_output.unsqueeze(0)
        days_output_list.append(days_stock_output)
        days_output = torch.cat(days_output_list, 0).permute(1, 0, 2)

        days_output = self.fc(days_output)

        # sent_output:
        return days_output

#
class Sent_Muil_Stock_HAN(nn.Module):
    def __init__(self, head_num,
                 days_num,
                 days_hidden_size,
                 news_hidden_size,
                 stock_hidden_size,
                 stock_length,
                 num_classes,
                 pretrained_word2vec_path,
                 dropout):
        super(Sent_Muil_Stock_HAN, self).__init__()
        self.head_num = head_num
        self.days_num = days_num
        self.days_hidden_size = days_hidden_size
        self.news_hidden_size = news_hidden_size
        self.stock_hidden_size = stock_hidden_size
        self.num_classes = num_classes
        self.pretrained_word2vec_path = pretrained_word2vec_path
        self.dropout = dropout
        self.days_news_att_net = Muil_DaysNewsAttNet(days_hidden_size, news_hidden_size, dropout=dropout)
        self.news_att_net = Muil_NewsAttNet(pretrained_word2vec_path, news_hidden_size, dropout=dropout)
        self.days_stock_att_net = Muil_DaysStockAttNet(days_hidden_size, stock_hidden_size, dropout=dropout)
        self.stock_att_net = Muil_StockAttNet(head_num=head_num, stock_length=stock_length, stock_hidden_size=stock_hidden_size,dropout=dropout)
        self.fc = nn.Linear(2 * days_hidden_size, num_classes, dropout=dropout)

    def forward(self, days_newses_input, days_stock_input):
        '''
        params:
            :param input: [batch_size, days_num, max_news_length, max_sent_length, max_word_length]
        return:
            :return output: []
        '''
        days_output_list = []
        # news net
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
            newses_output = torch.cat(news_output_list, 0).permute(1, 0, 2)
            # news attention net
            newses_output = self.news_att_net(newses_output)
            newses_output = newses_output.unsqueeze(0)
            days_output_list.append(newses_output)
        days_outputs = torch.cat(days_output_list, 0).permute(1, 0, 2)
        # newes_outputs: [batch_size, days_num, 2 * news_hidden_size]
        days_newses_outputs = self.days_att_net(days_outputs)

        # stock network
        days_stock_output_list = []
        days_stock_input = days_stock_input.permute(1, 0 ,2)
        for stock in days_stock_input:
            # stock: [batch_size, stock_length]
            stock_output = self.stock_att_net(stock)
            stock_output = stock_output.unsqueeze(0)
            days_stock_output_list.append(stock_output)
        days_stock_output = torch.cat(days_stock_output_list, 0).permute(1, 0, 2)
        days_stock_output = self.days_stock_att_net(days_stock_output)
        days_stock_output = days_stock_output.unsqueeze(0)
        days_output_list.append(days_stock_output)
        days_output = torch.cat(days_output_list, 0).permute(1, 0, 2)

        days_output = self.fc(days_output)

        # output:
        return days_output