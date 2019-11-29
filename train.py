"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.han_model import *
from src.stock_han_model import *
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    # training params
    parser.add_argument("--model_type", type=str, default="ori_han")    # model_type : ori_han; sent_ori_han; muil_han; sent_muil_han;muil_stock_han;sent_muil_stock_han
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # model params
    parser.add_argument("--add_stock", type=bool, default=False)
    parser.add_argument("--days_hidden_size", type=int, default=16)
    parser.add_argument("--news_hidden_size", type=int, default=8)
    parser.add_argument("--sent_hidden_size", type=int, default=4)
    parser.add_argument("--stock_hidden_size", type=int, default=16)
    parser.add_argument("--head_num", type=int, default=8)
    parser.add_argument("--days_num", type=int, default=12)
    # data params
    parser.add_argument("--train_set", type=str, default="/home/ingrid/Data/stockpredict_20191105/train_data.csv")
    parser.add_argument("--test_set", type=str, default="/home/ingrid/Data/stockpredict_20191105/test_data.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="/home/ingrid/Model/glove_ch/vectors_50.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # training setting
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    # training dataset info
    max_news_length, max_sent_length, max_word_length = get_max_lengths(opt.train_set)
    stock_length = 9
    training_set = MyDataset(data_path=opt.train_set,
                             dict_path=opt.word2vec_path,
                             max_news_length=max_news_length,
                             max_sent_length=max_sent_length,
                             max_word_length=max_word_length,
                             days_num=opt.days_num,
                             stock_length=stock_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(data_path=opt.test_set,
                         dict_path=opt.word2vec_path,
                         max_news_length=max_news_length,
                         max_sent_length=max_sent_length,
                         max_word_length=max_word_length,
                         days_num=opt.days_num,
                         stock_length=stock_length)
    test_generator = DataLoader(test_set, **test_params)

    # model init
    if opt.model_type == "ori_han":
        model = Ori_HAN(days_num=opt.days_num,
                        days_hidden_size=opt.days_hidden_size,
                        news_hidden_size=opt.news_hidden_size,
                        num_classes=training_set.num_classes,
                        pretrained_word2vec_path=opt.word2vec_path,
                        dropout=opt.dropout)
    elif opt.model_type == "sent_ori_han":
        model = Sent_Ori_HAN(days_num=opt.days_num,
                             days_hidden_size=opt.days_hidden_size,
                             news_hidden_size=opt.news_hidden_size,
                             sent_hidden_size=opt.sent_hidden_size,
                             num_classes=training_set.num_classes,
                             pretrained_word2vec_path=opt.word2vec_path,
                             dropout=opt.dropout)
    elif opt.model_type == "muil_han":
        model = Muil_HAN(head_num=opt.head_num,
                         days_num=opt.days_num,
                         days_hidden_size=opt.days_hidden_size,
                         news_hidden_size=opt.news_hidden_size,
                         num_classes=training_set.num_classes,
                         pretrained_word2vec_path=opt.word2vec_path,
                         dropout=opt.dropout)
    elif opt.model_type == "sent_muil_han":
        model = Sent_Muil_HAN(head_num=opt.head_num,
                              days_num=opt.days_num,
                              days_hidden_size=opt.days_hidden_size,
                              news_hidden_size=opt.news_hidden_size,
                              sent_hidden_size=opt.sent_hidden_size,
                              num_classes=training_set.num_classes,
                              pretrained_word2vec_path=opt.word2vec_path,
                              dropout=opt.dropout)
    elif opt.model_type == "muil_stock_han":
        model = Muil_Stock_HAN(head_num=opt.head_num,
                               days_num=opt.days_num,
                               days_hidden_size=opt.days_hidden_size,
                               news_hidden_size=opt.news_hidden_size,
                               stock_hidden_size=opt.stock_hidden_size,
                               stock_length=stock_length,
                               num_classes=training_set.num_classes,
                               pretrained_word2vec_path=opt.word2vec_path,
                               dropout=opt.dropout)
    elif opt.model_type == "sent_muil_stock_han":
        model = Sent_Muil_Stock_HAN(head_num=opt.head_num,
                                    days_num=opt.days_num,
                                    days_hidden_size=opt.days_hidden_size,
                                    news_hidden_size=opt.news_hidden_size,
                                    sent_hidden_size=opt.sent_hidden_size,
                                    stock_hidden_size=opt.stock_hidden_size,
                                    stock_length=stock_length,
                                    num_classes=training_set.num_classes,
                                    pretrained_word2vec_path=opt.word2vec_path,
                                    dropout=opt.dropout)

    # other setting
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path) # 递归删除文件夹下的所有子文件夹
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # 模型训练相关信息初始化
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    # 训练模型
    for epoch in range(opt.num_epoches):
        for iter, (days_news, days_stock, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                days_news = days_news.cuda()
                days_stock = days_stock.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            if opt.model_type in ["ori_han", "sent_ori_han", "muil_han", "sent_muil_han"]:
                predictions = model(days_news)
            elif opt.model_type in ["muil_stock_han", "sent_muil_stock_han"]:
                predictions = model(days_news, days_stock)
            loss = criterion(predictions, torch.tensor(label))
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_days_news, te_days_stock, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_days_news = te_days_news.cuda()
                    te_days_stock = te_days_stock.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    if opt.model_type == "ori_han":
                        te_predictions = model(te_days_news)
                te_loss = criterion(te_predictions, torch.tensor(te_label))
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + opt.model_type + "_model")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
