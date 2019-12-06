
import datetime
import pandas as pd
from src.utils import *

# 处理新闻文本数据
file_name = "news.csv"
news_df = pd.read_csv(data_root + file_name, sep=',', header=0).astype(str)

# 处理title和content数据
news_df['title'] = news_df['title'] + '。'
news_df['title'].replace('\s+', '，', regex=True, inplace=True)
news_df['content'].replace('\s+', '，', regex=True, inplace=True)
news_df['content'] = news_df['title'] + news_df['content']

# 合并同一日期新闻
dates = set(str(date) for date in news_df['date'])
dates_news_list = []
for date in dates:
    date_news_df = news_df[news_df['date'] == date]['content']
    date_news = ""
    for news in date_news_df:
        date_news = date_news + str(news) + '\t'
    date_news_dic = {'date': str(date), 'date_news': date_news}
    dates_news_list.append(date_news_dic)

# 填充为空的日期新闻
date_range = pd.date_range(start='20140414', end='20190401', freq='D')
for date in date_range:
    date = date.strftime("%Y%m%d")
    if date not in str(dates):
        date_news_dic = {'date': date, 'date_news': ""}
        dates_news_list.append(date_news_dic)

new_news_df = pd.DataFrame(dates_news_list)
new_news_df.sort_values(by='date', ascending=True, inplace=True)
new_news_df.drop(len(new_news_df)-1, axis=0, inplace=True)
new_news_df.reset_index(drop=True, inplace=True)
# new_news_df.to_csv(path_or_buf=data_root + "new_news.csv", sep=',',  index_label='index')
print("Success to handle content data...")





# 处理股票交易数据
# []
file_name = "shanghai_indics.csv"
min_date = '20140414'
max_date = '20190401'
# 基本处理
# columns:date,stock_code,stock_name,closing_price,top_price,low_price,open_price,close_price,ups_and_downs,Chg,volumns,AMO
stock_df = pd.read_csv(data_root+file_name, sep=',', header=0).astype(str)
stock_df.drop(["stock_name", "stock_code"], axis=1, inplace=True)
stock_df['date'] = stock_df['date'].apply(lambda x: x.replace("-", ''))
stock_df = stock_df[(stock_df['date']>=min_date) & (stock_df['date']<=max_date)]
# 填充缺失日期的数据
dates = set(str(date) for date in stock_df['date'])
numeric_columns = ['closing_price', 'top_price', 'low_price', 'open_price', 'close_price', 'ups_and_downs', 'Chg', 'volumns', 'AMO']
for column in numeric_columns:
    stock_df[column] = pd.to_numeric(stock_df[column])
date_range = pd.date_range(start='20140414', end='20190401', freq='D')
add_stock_list = []
for date in date_range:
    if date.strftime("%Y%m%d") not in str(dates):
        date_stock_dic = {}
        date_stock_dic['date'] = date.strftime("%Y%m%d")
        for column in numeric_columns:
            date_stock_dic[column] = stock_df[column].mean()
        add_stock_list.append(date_stock_dic)
add_stock_df = pd.DataFrame(add_stock_list)
add_stock_df['date'].astype(str)
stock_df = pd.concat([stock_df, add_stock_df])
# 填充为空的数据
# print(stock_df.columns[stock_df.isnull().sum() > 0])
# 添加label列
stock_df.sort_values(by='date', ascending=True, inplace=True)
date_range = pd.date_range(start='20140414', end='20190331', freq='D')
label_list = []
for date in date_range:
    date_label_dic = {}
    date_label_dic['date'] = date.strftime("%Y%m%d")
    date_stock_df = stock_df[stock_df['date'] == date.strftime("%Y%m%d")]['open_price']
    now_value = date_stock_df.iloc[0]
    date_stock_df = stock_df[stock_df['date'] == (date+datetime.timedelta(days=1)).strftime("%Y%m%d")]['open_price']
    next_value = date_stock_df.iloc[0]
    if float(next_value) > float(now_value):
        date_label_dic['label'] = 1
    else:
        date_label_dic['label'] = 0
    label_list.append(date_label_dic)
label_df = pd.DataFrame(label_list)
label_df['date'].astype(str)
label_df.sort_values(by='date', ascending=True, inplace=True)
# stock_df.drop(stock_df['date'] == '20190401', axis=0, inplace=True)
new_stock_df = pd.merge(label_df, stock_df, on='date')
# 数据标准化
for column in numeric_columns:
    new_stock_df[column] = (new_stock_df[column] - new_stock_df[column].min()) / (new_stock_df[column].max() - new_stock_df[column].min())

new_stock_df.sort_values(by='date', ascending=True, inplace=True)
new_stock_df.reset_index(drop=True, inplace=True)
# new_stock_df.to_csv(path_or_buf=data_root + "new_stock.csv", sep=',',  index_label='index')
print("Success to handle stock data...")

# 合并news和stock数据
data_df = pd.merge(new_stock_df, new_news_df, on='date')
print("Success to merge news and stock data...")
file_name = "new_data.csv"
data_df.to_csv(path_or_buf=data_root + file_name, sep=',', index_label='index')
print("Success to save data in file : "+data_root + file_name)


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
for index in range(len(total_newses)):
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
train_data_df = data_df[:int(0.7*len(data_df))]
train_data_df.to_csv(path_or_buf=data_root + "train_data_{}.csv".format(days_num), sep=',', index_label='index')
test_data_df = data_df[int(0.7*len(data_df)):]
train_data_df.to_csv(path_or_buf=data_root + "test_data_{}.csv".format(days_num), sep=',', index_label='index')
print("Success to split data and save them...")
