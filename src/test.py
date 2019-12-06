
import datetime
import pandas as pd
from src.utils import *

# 处理新闻文本数据

file_name = "new_data.csv"
data_df = pd.read_csv(data_root + file_name)


# 将数据处理为模型可以直接读取的结果
total_newses, total_stocks, total_labels = [], [], []
file_path = data_root+file_name
with open(data_root+file_name) as csv_file:
    has_header = csv.Sniffer().has_header(csv_file.read(1024))
    csv_file.seek(0)  # Rewind.
    reader = csv.reader(csv_file, quotechar='"')
    if has_header:
        next(reader)  # Skip header row.
    for idx, line in enumerate(reader):
        news = str(line[-1])
        stock = [float(x) for x in line[3:-1]]
        label = int(line[2])
        total_newses.append(news)
        total_stocks.append(stock)
        total_labels.append(label)
    print("Success to load the data...")
    dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0], engine='python').values
    dict = [word[0] for word in dict]
    print("Success to build the dic...")
    max_news_length, max_sent_length, max_word_length = get_max_lengths(data_root+file_name)
    print("Success to get the length info about the data...")
    num_classes = len(set(total_labels))

total_data = []
for index in range(2):
    data_init_time = datetime.datetime.now()
    label = total_labels[index]
    if index < 12:
        days_newses = total_newses[0: index + 1]
        days_stock = total_stocks[0: index + 1]
    else:
        days_newses = total_newses[index - days_num + 1: index + 1]
        days_stock = total_stocks[index - days_num + 1: index + 1]

    # prepare news data
    days_newses_encode = [[[[dict.index(word) if word in dict else -1
                             for word in cut(sent)]
                            for sent in news.split("。")]
                           for news in newses.split("\t")]
                          for newses in days_newses]  # 对文段中的每一个word标记其在dict中的index

    for newses in days_newses_encode:
        for news in newses:
            for sent in news:
                if len(sent) < max_word_length:
                    extended_words = [-1 for _ in range(max_word_length - len(sent))]
                    sent.extend(extended_words)
            if len(news) < max_sent_length:
                extended_sentence = [[-1 for _ in range(max_word_length)]
                                     for _ in range(max_sent_length - len(news))]
                news.extend(extended_sentence)
        if len(newses) < max_news_length:
            extended_news = [[[-1 for _ in range(max_word_length)]
                              for _ in range(max_sent_length)]
                             for _ in range(max_news_length - len(newses))]
            newses.extend(extended_news)

    if len(days_newses_encode) < days_num:
        extended_days_newses = [[[[-1 for _ in range(max_word_length)]
                                  for _ in range(max_sent_length)]
                                 for _ in range(max_news_length)]
                                for _ in range(days_num - len(days_newses_encode))]
        days_newses_encode.extend(extended_days_newses)

    new_days_newses_encode = []
    for newses in days_newses_encode:
        new_newses = []
        for news in newses:
            new_news = []
            for sent in news:
                new_sent = sent[:max_word_length]
                new_sent = [x+1 for x in new_sent]
                new_news.append(new_sent)
            new_newses.append(new_news[:max_sent_length])
        new_days_newses_encode.append(new_newses[:max_news_length])
    days_newses_encode = new_days_newses_encode

    # document_encode = [sentence[:self.max_length_word] for sentence in document_encode][:self.max_length_sentence]
    # days_newses_encode = np.stack(arrays=days_newses_encode, axis=0)

    # days_newses_encode += 1

    # prepare stock date
    if len(days_stock) < days_num:
        extended_stock = [[-1 for _ in range(stock_length)]
                      for _ in range(days_num - len(days_stock))]
        days_stock.extend(extended_stock)
    # days_stock = np.stack(days_stock, axis=0)

    # per_data = {"index": index, "per_news_data": days_newses_encode.astype(np.int64), "per_stock_data": days_stock.astype(np.float32), "label": label}
    per_data = {"index": index, "per_news_data": days_newses_encode, "per_stock_data": days_stock, "label": label}
    total_data.append(per_data)
    data_end_time = datetime.datetime.now()
    print("the handling time for index {} is: {}s...".format(index, (data_end_time - data_init_time).seconds))
    print("Success to preprocess the data of index : " + str(index))
data_df = pd.DataFrame(total_data)
data_df.sort_values(by='index', ascending=True, inplace=True)
print("Success to get the data for input...")

# 将数据分为训练数据集和测试数据集
# train_data_df = data_df[:int(0.7*len(data_df))]
data_df.to_csv(path_or_buf=data_root + "train_data_{}.csv".format(days_num), sep=',', index_label='index')
print("Success to save data preprocessed...")
# test_data_df = data_df[int(0.7*len(data_df)):]
# train_data_df.to_csv(path_or_buf=data_root + "test_data_{}.csv".format(days_num), sep=',', index_label='index')
# print("Success to split data and save them...")

data_df = pd.read_csv(data_root + file_name)
