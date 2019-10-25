import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def train():
    print("开始训练...")


def evaluate():
    print("开始验证...")


if __name__ == '__main__':
    print("加载数据")
    print("创建模型")
    print("定义损失")
    print("定义优化器")
