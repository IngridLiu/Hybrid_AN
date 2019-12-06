from src.utils import *

train_data_df = pd.read_csv
train_data_df.to_csv(path_or_buf=data_root + "train_data_{}.csv".format(days_num), sep=',', index_label='index')
test_data_df = data_df[int(0.7*len(data_df)):]
train_data_df.to_csv(path_or_buf=data_root + "test_data_{}.csv".format(days_num), sep=',', index_label='index')
print("Success to split data and save them...")