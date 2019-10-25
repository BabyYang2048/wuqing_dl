import torch
import torch.nn as nn
from torch.nn import functional as F
from trans_att.model import Encoder, Decoder, padding_mask
from trans_att.my_dataset import get_train_dataloader,get_val_dataloader,get_test_dataloader,get_ipt_vocab_size,get_opt_vocab_size
import pytorch_lightning as pl
from pytorch_lightning import Trainer


src_vocab_size = get_ipt_vocab_size()
src_max_len = 15
tgt_vocab_size = get_opt_vocab_size()
tgt_max_len = 15


####################################################################
# Transformer
####################################################################
class TransformerModel(pl.LightningModule):

    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)
        output, enc_self_attn = self.encoder(src_seq, src_len)
        output, dec_self_attn, ctx_attn = self.decoder(
          tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn

    def training_step(self, batch, batch_nb):
        """
        返回损失，附带tqdm指标的dict
        :param batch:
        :param batch_nb:
        :return:
        """
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)  #y_hat对应为线性回归模型的预测值
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        """
        返回需要在validation_end中聚合的所有输出
        :param batch:
        :param batch_nb:
        :return:
        """
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        train_data = get_train_dataloader()
        return train_data

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        val_data = get_val_dataloader()
        return val_data

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        test_data = get_test_dataloader()
        return test_data


model = TransformerModel(src_vocab_size,src_max_len,tgt_vocab_size,tgt_max_len);

# most basic trainer, uses good defaults
trainer = Trainer(gpus='0', max_nb_epochs=3)
trainer.fit(model)