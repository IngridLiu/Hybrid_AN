
import time
import pandas as pd
# []

data_root = "/home/ingrid/Data/stockpredict_20191105/"

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
new_news_df.reset_index(drop=True, inplace=True)
print(new_news_df.columns)
new_news_df.to_csv(path_or_buf=data_root + "new_news.csv", sep=',',  index_label='index')












# 处理股票交易数据