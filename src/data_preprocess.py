
import datetime
import pandas as pd
import numpy as np
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
new_news_df.drop(len(new_news_df)-1, axis=0, inplace=True)
new_news_df.reset_index(drop=True, inplace=True)
new_news_df.to_csv(path_or_buf=data_root + "new_news.csv", sep=',',  index_label='index')
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
# 填充缺失数据
dates = set(str(date) for date in stock_df['date'])
numeric_columns = ['closing_price', 'top_price', 'low_price' ,'open_price' ,'close_price' ,'ups_and_downs' ,'Chg' ,'volumns' ,'AMO']
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
new_stock_df.to_csv(path_or_buf=data_root + "new_stock.csv", sep=',',  index_label='index')

print(new_stock_df.columns)
print(new_stock_df.head(5))
print("Success to handle stock data...")