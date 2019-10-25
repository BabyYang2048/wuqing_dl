import torch
from torch.utils.data import Dataset, DataLoader
import pickle

i = 9
vocab_data_path = "../data/vocab.pkl"
train_data_path = "../data/train_data_"+str(i)+".pkl"
valid_data_path = "../data/valid_data_"+str(i)+".pkl"
test_data_path = "../data/test_data.pkl"


class DiseaseQuestion(Dataset):
    def __init__(self, data, vocab, max_len=15):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def pad_and_clip(self, sen):
        pad_len = self.max_len - len(sen)
        if pad_len > 0:
            return sen + [self.vocab.get("PAD")] * pad_len
        else:
            return sen[:self.max_len]

    # def trans_sent2id(self, sen):
    #     res = []
    #     for c in sen:
    #         # try:
    #         #     res.append(self.vocab.get(c))
    #         # except:
    #         #     res.append(self.vocab.get('UNK'))
    #         if self.vocab.get(c) is None:
    #             res.append(self.vocab.get('UNK'))
    #         else:
    #             res.append(self.vocab.get(c))
    #     # [self.vocab.get(c) for c in sen]
    #     # 这里得到一个点：对于python，get字典里的值get不到的时候是返回的None，而不是抛出异常
    #     return res

    # 这个方法里面return的值等于上面注释的方法里面的一长段
    def trans_sent2id(self, sen):
        return [self.vocab.get(c, 1) for c in sen]

    def __getitem__(self, item):
        sen1, sen2, label = self.data[item]
        sen1 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen1))).long()
        sen2 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen2))).long()
        return sen1, sen2, torch.tensor(label)

    def __len__(self):
        return len(self.data)


vocab_data = pickle.load(open(vocab_data_path,'rb'))


# 默认 64
def get_train_dataloader(batch_size=64):
    train_data = pickle.load(open(train_data_path, 'rb'))
    train_loader = DataLoader(DiseaseQuestion(train_data, vocab_data), batch_size=batch_size, shuffle=True)
    return train_loader


def get_valid_dataloader(batch_size=64):
    valid_data = pickle.load(open(valid_data_path, 'rb'))
    valid_loader = DataLoader(DiseaseQuestion(valid_data, vocab_data), batch_size=batch_size, shuffle=True)
    return valid_loader


def get_test_dataloader(batch_size=64):
    test_data = pickle.load(open(test_data_path,'rb'))
    test_loader = DataLoader(DiseaseQuestion(test_data, vocab_data), batch_size=batch_size, shuffle=True)
    return test_loader

