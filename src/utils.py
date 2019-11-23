"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from sklearn import metrics
from jieba import cut
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def get_max_lengths(data_path):
    news_length_list = []
    sent_length_list = []
    word_length_list = []
    with open(data_path) as csv_file:
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)  # Rewind.
        reader = csv.reader(csv_file, quotechar='"')
        if has_header:
            next(reader)  # Skip header row.
        for idx, line in enumerate(reader):
            news_list = line[-1].split("\t")
            news_length_list.append(len(news_list))

            for news in news_list:
                sent_list = news.split("。")
                sent_length_list.append(len(sent_list))

                for sent in news.split("。"):
                    word_list = list(cut(sent))
                    word_length_list.append(len(word_list))

        sorted_news_length = sorted(news_length_list)
        sorted_sent_length = sorted(sent_length_list)
        sorted_word_length = sorted(word_length_list)

    return sorted_news_length[int(0.8*len(sorted_news_length))], \
           sorted_sent_length[int(0.8 * len(sorted_sent_length))], \
           sorted_word_length[int(0.8*len(sorted_word_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)






