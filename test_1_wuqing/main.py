import torch
import time
import torch.nn as nn
from test_1_wuqing.models import BaseClassification
from test_1_wuqing.my_dataset import get_train_dataloader, get_test_dataloader, get_valid_dataloader
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

i = 9


def data_to_device(data):
    for d in data:
        yield d.to(device)


def train(model, train_loader, valid_loader, test_loader, loss_func, optimizer, epochs=5):
    best_score = 0.5
    for ep in range(epochs):
        print("EPOCH", ep)
        acc_num = 0
        losses = []
        start_time = time.time()
        for ii, d in enumerate(train_loader):
            x1, x2, label = data_to_device(d)  # 把数据取出来
            optimizer.zero_grad()  # 整个网络梯度归0
            output = model(x1, x2)  # 得到y'
            loss = loss_func(output, label)  # 计算损失值
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            losses.append(loss.cpu().item())
            acc_num += (torch.argmax(output, dim=-1).long() == label.long()).sum().cpu().item()
        print("EPOCH %d, TRAIN MEAN LOSS = %f, TRAIN ACCURACY = %f, SPEND TIME: %d" %
              (ep, sum(losses) / len(losses), acc_num / len(train_loader.dataset), time.time() - start_time))
        score = evaluate(model, valid_loader, ep)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), '../models/model_'+str(i)+'.pkl')
        print("BEST SCORE:", best_score)


def evaluate(model, valid_loader, ep):
    model.eval()
    with torch.no_grad():
        acc_num = 0
        losses = []
        for ii, d in enumerate(valid_loader):
            x1, x2, label = data_to_device(d)
            output = model(x1, x2)
            # 这里如果预测不出来的话一般是数据格式的问题
            # 好的，我们下一步来看看数据格式
            # print(output)
            loss = loss_func(output, label)
            losses.append(loss.cpu().item())
            acc_num += (torch.argmax(output, dim=-1).long() == label.long()).sum().cpu().item()
        print("EPOCH %d, VALID MEAN LOSS = %f, VALID ACCURACY = %f" %
              (ep, sum(losses) / len(losses), acc_num / len(valid_loader.dataset)))
    model.train()
    return acc_num / len(valid_loader.dataset)


def evaluate3(models, valid_loader):
    acc_num = 0
    for ii, d in enumerate(valid_loader):
        x1, x2, label = data_to_device(d)
        final_out = torch.zeros((x1.size(0), 2)).to(device)
        for model in models:
            output = model(x1, x2)
            final_out += output
        acc_num += (torch.argmax(final_out, dim=-1).long() == label.long()).sum().cpu().item()
    print("VALID ACCURACY = %f" % (acc_num / len(valid_loader.dataset)))
    return acc_num / len(valid_loader.dataset)


def get_p_r_f1(y_s, y_hats):
    p = precision_score(y_s, y_hats, pos_label=1)
    r = recall_score(y_s, y_hats, pos_label=1)
    f1 = f1_score(y_s, y_hats, pos_label=1)
    return p, r, f1


if __name__ == '__main__':
    # 加载数据
    print("加载数据")
    train_loader = get_train_dataloader()
    valid_loader = get_valid_dataloader()
    test_loader = get_test_dataloader()
    is_train = False
    if is_train:
        # 创建模型
        print("创建模型")
        model = BaseClassification(len(train_loader.dataset.vocab), 128, mode="lstm")
        model.to(device)
        # 定义损失函数
        print("定义损失")
        loss_func = nn.CrossEntropyLoss()
        # 定义优化器
        print("定义优化器")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        # 训练模型
        print("训练模型")
        train(model, train_loader, valid_loader, test_loader, loss_func, optimizer, epochs=15)
    else:
        models = []
        models_path = ['../models/model_1.pkl', '../models/model_2.pkl', '../models/model_3.pkl','../models/model_4.pkl','../models/model_5.pkl',
                       '../models/model_6.pkl','../models/model_7.pkl','../models/model_8.pkl','../models/model_9.pkl']
        for i in range(5, 9):
            model = BaseClassification(len(train_loader.dataset.vocab), 128, mode="lstm")
            model.load_state_dict(torch.load(models_path[i]))
            model.eval()
            model.to(device)
            models.append(model)
        evaluate3(models, test_loader)
        # model_1 0.74
        # model_1 -> model_2 0.783
        # model_1 -> model_3 0.7898
        # model_1 -> model_4 0.811947
        # model_1 -> model_9 0.831858
