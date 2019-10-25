import torch
from torch.utils.data import Dataset, DataLoader
from trans_att.data_prepare import add_vocab

first_path = "../data/first.txt"
second_path = "../data/second.txt"

ipt_vocab, opt_vocab,sen_pairs = add_vocab(first_path,second_path)


class MyDataset(Dataset):
    def __init__(self, data, vocab_ipt, vocab_opt, max_len=15):
        self.data = data
        self.vocab_ipt = vocab_ipt
        self.vocab_opt = vocab_opt
        self.max_len = max_len

    def pad_and_clip(self, sen):
        pad_len = self.max_len - len(sen)
        if pad_len > 0:
            return sen + [self.vocab_ipt.get("PAD")] * pad_len
        else:
            return sen[:self.max_len]

    def sen2id_ipt(self, sen):
        return self.tans_sent2id(self.vocab_ipt,sen)

    def sen2id_opt(self, sen):
        return self.tans_sent2id(self.vocab_opt,sen)

    def tans_sent2id(self, vocab, sen):
        return [vocab.get(c) for c in sen.replace(" ","")]

    def __getitem__(self, item):
        ipt = self.data[item][0]
        opt = self.data[item][1]
        ipt = torch.tensor(self.pad_and_clip(self.sen2id_ipt(ipt))).long()
        opt = torch.tensor(self.pad_and_clip(self.sen2id_opt(opt))).long()
        return ipt, opt

    def __len__(self):
        return len(self.data)


def get_train_dataloader(batch_size=64):
    train_data = sen_pairs
    train_loader = DataLoader(MyDataset(train_data, ipt_vocab, opt_vocab), batch_size=batch_size, shuffle=False)
    return train_loader


def get_val_dataloader(batch_size=64):
    val_data = sen_pairs
    val_loader = DataLoader(MyDataset(val_data, ipt_vocab, opt_vocab), batch_size=batch_size, shuffle=False)
    return val_loader


def get_test_dataloader(batch_size=64):
    test_data = sen_pairs
    test_loader = DataLoader(MyDataset(test_data, ipt_vocab, opt_vocab), batch_size=batch_size, shuffle=False)
    return test_loader


def get_ipt_vocab_size():
    return ipt_vocab.__len__()


def get_opt_vocab_size():
    return opt_vocab.__len__()