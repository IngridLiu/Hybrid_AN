"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation, get_max_lengths
from src.dataset import MyDataset
import argparse
import shutil
import csv
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--model_type", type=str, default="ori_han")  # model_type : ori_han; sent_ori_han; muil_han; sent_muil_han;muil_stock_han;sent_muil_stock_han
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_set", type=str, default="/home/ingrid/Data/stockpredict_20191105/train_data.csv")
    parser.add_argument("--test_set", type=str, default="/home/ingrid/Data/stockpredict_20191105/test_data.csv")
    parser.add_argument("--model_path", type=str, default="/home/ingrid/Projects/PythonProjects/stock_predict/trained_models")
    parser.add_argument("--word2vec_path", type=str, default="/home/ingrid/Model/glove_ch/vectors_50.txt")
    parser.add_argument("--output", type=str, default="/home/ingrid/Projects/PythonProjects/predictions")
    args = parser.parse_args()
    return args


def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        model = torch.load(opt.model_path + os.sep + opt.model_type + "_model")
    else:
        model = torch.load(opt.model_path + os.sep + opt.model_type + "_model", map_location=lambda storage, loc: storage)
    max_news_length, max_sent_length, max_word_length = get_max_lengths(opt.train_set)
    stock_length = 9
    test_set = MyDataset(data_path=opt.test_set,
                         dict_path=opt.word2vec_path,
                         max_news_length=max_news_length,
                         max_sent_length=max_sent_length,
                         max_word_length=max_word_length,
                         days_num=opt.days_num,
                         stock_length=stock_length)
    test_generator = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    for te_days_news, te_days_stock, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_days_news.cuda()
            te_days_stock = te_days_stock.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            if opt.model_type in ["ori_han", "sent_ori_han", "muil_han", "sent_muil_han"]:
                te_predictions = model(te_days_news)
            elif opt.model_type in ["muil_stock_han", "sent_muil_stock_han"]:
                te_predictions = model(te_days_news, te_days_stock)
            te_predictions = F.softmax(te_predictions)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)

    fieldnames = ['True label', 'Predicted label', 'Content']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(te_label, te_pred, test_set.newses, test_set.stocks):
            writer.writerow(
                {'True label': i + 1, 'Predicted label': np.argmax(j) + 1, 'Content': k})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
