import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


####################################################################
# word-embedding 词向量转换，这个在encoder和decoder中写了~就不单独封装了
####################################################################
# embedding = nn.Embedding(vocab_size,embedding_size,padding_idx=0)
# seq_embedding = seq_embedding(inputs) * np.sqrt(d_model)


####################################################################
# Positional-encoding 存储句子的位置特征
####################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """
        初始化
        :param d_model:模型的维度，默认512
        :param max_seq_len:文本序列最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给出的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)
        ])

        # 偶数列是sin 奇数列是cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这'PAD'的positional encoding
        # 在word embedding中也经常加上'UNK'代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model]).float()
        position_encoding = torch.cat((pad_row, torch.tensor(position_encoding).float()))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len+1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,requires_grad=False)

    def forward(self, input_len):
        """
        正向传播
        :param input_len: 形状为[batch_size, 1] 每一个张量的值都代表这一批文本序列对应的长度
        :return: 返回这一批序列的位置编码，进行了对齐
        """
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor([list(range(1,len+1))+[0]*(max_len-len) for len in max_len])
        return self.position_encoding(input_pos)


####################################################################
# Scaled Dot-Production Attention 点积注意力机制
####################################################################
class ScaledDotProductionAttention(nn.Module):
    def __init__(self,attention_dropout):
        super(ScaledDotProductionAttention,self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,q,k,v,scale=None,attn_mask=None):
        """
        向前传播
        :param q: queries张量,形状为[B,L_q,D_q]
        :param k: keys张量，形状为[B,L_k,D_k]
        :param v: values张量，形状为[B,L_v,D_v]
        :param scale: 缩放因子，是一个浮点标量，1/sqrt(D_k)
        :param attn_mask: Masking张量，形状[B,L_q,L_k]
        :return:上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1,2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention,v)
        return context, attention


####################################################################
# residual 残差连接Add
####################################################################
def residual(sublayer_fn,x):
    return sublayer_fn(x) + x


####################################################################
# layer normalization 层归一化
####################################################################


####################################################################
# Multi-Head Attention 基于自注意力！
####################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self,model_dim=512,num_heads=8,dropout=0.0):
        super(MultiHeadAttention,self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head* num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head* num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head* num_heads)

        self.dot_product_attention = ScaledDotProductionAttention(dropout)
        self.linear_final = nn.Linear(model_dim,model_dim)
        self.dorpout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        """
        向前传播
        :param key:
        :param value:
        :param query:
        :param attn_mask:
        :return:
        """
        residual = query  # 残差连接

        dim_pre_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size*num_heads,-1,dim_pre_head)
        value = value.view(batch_size*num_heads,-1,dim_pre_head)
        query = query.view(batch_size*num_heads,-1,dim_pre_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads,1,1)

        scale = (key.size(-1)//num_heads)**-0.5
        context,attention = self.dot_product_attention(query,key,value,scale,attn_mask)

        context = context.view(batch_size,-1,dim_pre_head*num_heads)
        output = self.linear_final(context)

        output = self.dorpout(output)

        output = self.layer_norm(residual + output)

        return output, attention


####################################################################
# padding-mask 针对长度补齐操作的掩码
####################################################################
def padding_mask(seq_k,seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1,len_q,-1)
    return pad_mask


####################################################################
# sequence-mask 针对decoder中的不关注后文的掩码
####################################################################
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len,seq_len),dtype=torch.uint8),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size,-1,-1)
    return mask


####################################################################
# positional wise feed forward 前向反馈，是一个全连接网络
####################################################################
class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


####################################################################
# Encoder
####################################################################
class EncoderLayer(nn.Module):
    """这是encoder的一层，共6层"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    """多层encoder_layer组成完整的encoder"""
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


####################################################################
# Decoder
####################################################################
class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask
            )
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions

