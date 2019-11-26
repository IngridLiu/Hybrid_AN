"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from jieba import cut
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_news_length=10, max_sent_length=10, max_word_length=10, days_num = 12):
        super(MyDataset, self).__init__()

        newses, stocks, labels = [], [], []
        with open(data_path) as csv_file:
            has_header = csv.Sniffer().has_header(csv_file.read(1024))
            csv_file.seek(0)  # Rewind.
            reader = csv.reader(csv_file, quotechar='"')
            if has_header:
                next(reader)  # Skip header row.
            for idx, line in enumerate(reader):
                news = str(line[-1])
                stock = [float(x) for x in line[3:-2]]
                label = int(line[2])
                newses.append(news)
                stocks.append(stock)
                labels.append(label)

        self.newses = newses
        self.stocks = stocks
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_news_length = max_news_length
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.days_num = days_num
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    # 获取model的单个输入和label
    def __getitem__(self, index):
        label = self.labels[index]
        days_num = self.days_num
        if index < 12:
            days_newses = self.newses
            days_stock = self.stocks[0: index+1]
        else:
            days_newses = self.newses[index-days_num+1: index+1]
            days_stock = self.stocks[index-days_num+1: index+1]

        # prepare news data
        days_newses_encode = [[[[self.dict.index(word) if word in self.dict else -1
             for word in cut(sent)]
            for sent in news.split("。")]
            for news in newses.split("\t")]
            for newses in days_newses] # 对文段中的每一个word标记其在dict中的index

        for newses in days_newses_encode:
            for news in newses:
                for sent in news:
                    if len(sent) < self.max_word_length:
                        extended_words = [-1 for _ in range(self.max_word_length - len(sent))]
                        sent.extend(extended_words)
                if len(news) < self.max_sent_length:
                    extended_sentence = [[-1 for _ in range(self.max_word_length)]
                                         for _ in range(self.max_sent_length-len(news))]
                    news.extend(extended_sentence)
            if len(newses) < self.max_news_length:
                extended_news = [[[-1 for _ in range(self.max_word_length)]
                                  for _ in range(self.max_sent_length)]
                                 for _ in range(self.max_news_length - len(newses))]
                newses.extend(extended_news)

        if len(days_newses_encode) < days_num:
            extended_days_newses = [[[[-1 for _ in range(self.max_word_length)]
                                  for _ in range(self.max_sent_length)]
                                 for _ in range(self.max_news_length)]
                               for _ in range(days_num - len(days_newses_encode))]
            days_newses_encode.extend(extended_days_newses)

        new_days_newses_encode = []
        for newses in days_newses_encode:
            new_newses = []
            for news in newses:
                new_news = []
                for sent in news:
                    new_sent = sent[:self.max_word_length]
                    new_news.append(new_sent)
                new_newses.append(new_news[:self.max_sent_length])
            new_days_newses_encode.append(new_newses[:self.max_news_length])
        days_newses_encode = new_days_newses_encode

        # document_encode = [sentence[:self.max_length_word] for sentence in document_encode][:self.max_length_sentence]
        days_newses_encode = np.stack(arrays=days_newses_encode, axis=0)
        days_newses_encode += 1

        # prepare stock date
        if len(days_stock) < days_num:
            stock_length = len(days_stock[0])
            extended_stock = [[-1 for _ in range(stock_length)]
                              for _ in range(days_num - len(days_stock))]
            days_stock.extend(extended_stock)
        days_stock = np.stack(days_stock, axis=0)
        days_stock += 1

        return days_newses_encode.astype(np.int64), days_stock.astype(np.float32), label

if __name__ == '__main__':
    test = MyDataset(data_path="../data/test.csv", dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
