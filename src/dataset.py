"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import ast
import json
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from jieba import cut
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path):
        super(MyDataset, self).__init__()

        data_df = pd.read_csv(data_path, header=0)
        # index, index, label, per_news_data, per_stock_data:
        data_df.sort_values(by="index", inplace=True)
        # print(data_df.columns)
        total_newses = data_df["per_news_data"].tolist()
        total_stocks = data_df["per_stock_data"].tolist()
        total_labels = data_df["label"].tolist()

        # with open(data_path) as csv_file:
        #     has_header = csv.Sniffer().has_header(csv_file.read(1024))
        #     csv_file.seek(0)  # Rewind.
        #     reader = csv.reader(csv_file, quotechar='"')
        #     if has_header:
        #         next(reader)  # Skip header row.
        #     for idx, line in enumerate(reader):
        #         days_newses = np.array(line[1]).astype(np.int64)
        #         days_stocks = np.array(line[0]).astype(np.float32)
        #         label = int(line[2])
        #         total_newses.append(days_newses)
        #         total_stocks.append(days_stocks)
        #         total_labels.append(label)

        self.total_newses = total_newses
        self.total_stocks = total_stocks
        self.total_labels = total_labels
        self.num_classes = len(set(self.total_labels))

    def __len__(self):
        return len(self.total_labels)

    # 获取model的单个输入和label
    def __getitem__(self, index):
        per_label = self.total_labels[index]
        per_newses = self.total_newses[index]
        per_stocks = self.total_stocks[index]
        # if index < 12:
        #     days_newses = self.newses[0: index+1]
        #     days_stock = self.stocks[0: index+1]
        # else:
        #     days_newses = self.newses[index-days_num+1: index+1]
        #     days_stock = self.stocks[index-days_num+1: index+1]
        #
        # # prepare news data
        # days_newses_encode = [[[[self.dict.index(word) if word in self.dict else -1
        #      for word in cut(sent)]
        #     for sent in news.split("。")]
        #     for news in newses.split("\t")]
        #     for newses in days_newses] # 对文段中的每一个word标记其在dict中的index
        #
        # for newses in days_newses_encode:
        #     for news in newses:
        #         for sent in news:
        #             if len(sent) < self.max_word_length:
        #                 extended_words = [-1 for _ in range(self.max_word_length - len(sent))]
        #                 sent.extend(extended_words)
        #         if len(news) < self.max_sent_length:
        #             extended_sentence = [[-1 for _ in range(self.max_word_length)]
        #                                  for _ in range(self.max_sent_length-len(news))]
        #             news.extend(extended_sentence)
        #     if len(newses) < self.max_news_length:
        #         extended_news = [[[-1 for _ in range(self.max_word_length)]
        #                           for _ in range(self.max_sent_length)]
        #                          for _ in range(self.max_news_length - len(newses))]
        #         newses.extend(extended_news)
        #
        # if len(days_newses_encode) < days_num:
        #     extended_days_newses = [[[[-1 for _ in range(self.max_word_length)]
        #                           for _ in range(self.max_sent_length)]
        #                          for _ in range(self.max_news_length)]
        #                        for _ in range(days_num - len(days_newses_encode))]
        #     days_newses_encode.extend(extended_days_newses)
        #
        # new_days_newses_encode = []
        # for newses in days_newses_encode:
        #     new_newses = []
        #     for news in newses:
        #         new_news = []
        #         for sent in news:
        #             new_sent = sent[:self.max_word_length]
        #             new_news.append(new_sent)
        #         new_newses.append(new_news[:self.max_sent_length])
        #     new_days_newses_encode.append(new_newses[:self.max_news_length])
        # days_newses_encode = new_days_newses_encode
        #
        # # document_encode = [sentence[:self.max_length_word] for sentence in document_encode][:self.max_length_sentence]
        # days_newses_encode = np.stack(arrays=days_newses_encode, axis=0)
        # days_newses_encode += 1
        #
        # # prepare stock date
        # if len(days_stock) < days_num:
        #     extended_stock = [[-1 for _ in range(self.stock_length)]
        #                       for _ in range(days_num - len(days_stock))]
        #     days_stock.extend(extended_stock)
        # days_stock = np.stack(days_stock, axis=0)
        #
        # return days_newses_encode.astype(np.int64), days_stock.astype(np.float32), label

        # list_list = ast.literal_eval(str_list)

        per_newses = ast.literal_eval(per_newses)
        per_newses = np.stack(per_newses, axis=0)
        per_stocks = ast.literal_eval(per_stocks)
        per_stocks = np.stack(per_stocks, axis=0)

        return per_newses.astype(np.int64), per_stocks.astype(np.float32), per_label

if __name__ == '__main__':
    test = MyDataset(data_path="/home/ingrid/Data/stockpredict_20191105/train_data.csv")
    print(len(test))
    shape_list = []
    for idx in range(len(test)):
        shape = test.__getitem__(idx)[0].shape
        if shape not in shape_list:
            shape_list.append({"index":idx, "shape":shape})
    print(shape_list)

