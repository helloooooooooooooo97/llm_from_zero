| 阶段| 完成 |章节 | 文章  | 备注| 练习 |
|---|---|---|-------|---|--|
|L1| ❓ | 🧊 算力 | [算力的统计与单位](./算力的单位与统计.md)  | 什么是算力的单位与统计，算力的单位与统计的原理是什么 | |
|L1| ❓ | 🧊 张量 | [张量的归约](./张量的归约.md)  | 什么是算力的单位与统计，算力的单位与统计的原理是什么 | |
|L1| ❓ | 🧊 张量 | [张量的广播](./张量的广播.md)  | 什么是算力的单位与统计，算力的单位与统计的原理是什么 | |
|L1| ✅ | 🧊 张量 | [张量的连续性](./张量的连续性.md)  | 张量的连续性是什么，为什么需要张量的连续性 | |
|L1| ✅ | 🧊 层归一化 | [层归一化](./层归一化.md)  | transformer的层归一化是什么，为什么需要层归一化 | 🔥[layer_norm.py](exercise/layer_norm.py)|
|L1| ✅ | 🧊 层归一化 | [为什么NLP要用层归一化而不是批归一化呢](./为什么NLP要用层归一化而不是批归一化呢.md)  | 为什么NLP要用层归一化而不是批归一化呢，层归一化与批归一化的区别是什么 | |
|L1| ✅ | 🧊 层归一化 | [Pre-LN和Post-LN对比](./Pre-LN和Post-LN对比.md)  | transformer的Pre-LN和Post-LN是什么，为什么需要Pre-LN和Post-LN | |
|L1| ✅ | 🔥 层归一化 | [层归一化代码与参数量](./层归一化代码与参数量.md) | 层归一化代码与参数量是什么，层归一化代码与参数量的原理是什么 | 🔥[layer_norm.py](exercise/layer_norm.py)|
|L1| ✅ | 🧊 dropout | [dropout的编码器位置](./dropout.md)  | 什么是dropout，dropout的原理是什么 | 🔥[dropout.py](exercise/dropout.py)
|L1| ✅ | 🧊 自注意力机制 | [Transformer](./transformer.md)  | 什么是Transformer，Transformer的原理是什么 | 🔥[transformer.py](exercise/transformer.py)|
|L1| ✅ | 🧊 自注意力机制 | [自注意力机制](./自注意力机制.md)  | 什么是自注意力机制，自注意力机制的原理是什么 | 🔥[self_attention.py](exercise/self_attention.py)|
|L1| ✅ | 🧊 自注意力机制 | [缩放因子的三个作用](./缩放因子的三个作用.md)  | 为什么需要缩放因子，缩放因子的作用是什么 | |
|L1| ✅ | 🧊 自注意力机制 | [残差连接](./残差连接.md)  | transformer的残差连接是什么，为什么需要残差连接 | |
|L1| ✅ | 🧊 注意力机制 | [多头注意力机制](./多头注意力机制.md)  | 为什么需要多头注意力机制，什么是多头注意力机制，多头注意力机制的原理是什么 | 🔥[exercise/multi_head_attention.py](exercise/multi_head_attention.py)|
|L1| ✅ | 🧊 位置编码 | [Transformer的位置编码](./Transformer的位置编码.md)  | 理解为什么需要位置编码以及数字位置编码的不足 |
|L1| ✅ | 🧊 位置编码 | [理想Transformer位置编码的五大特性](./理想Transformer位置编码的五大特性.md)  | 理解一个好的位置编码需要满足的特性是什么 |
|L1| ✅ | 🧊 位置编码 | [绝对位置编码如何满足五大特性](./绝对位置编码如何满足五大特性.md)  | 理解SINUSOIDAL POSITION ENCODING是如何满足位置编码的特性 |
|L1| ✅ | 🧊 位置编码 | [绝对位置编码中的相对位置信息如何体现的呢](./绝对位置编码中的相对位置信息如何体现的呢.md) | 理解为什么绝对位置编码中编码了相对位置信息 |
|L1| ✅ | 🧊 位置编码 | [绝对位置编码的相对位置信息的证明](./绝对位置编码的相对位置信息的证明.md) | 证明绝对位置编码中编码了相对位置信息 |
|L1| ✅ | 🧊 位置编码 | [旋转位置编码如何编码相对位置信息](./旋转位置编码如何编码相对位置信息.md) | 旋转位置编码的如何直接编码相对位置信息 |
|L1| ✅ | 🧊 位置编码 | [旋转矩阵的定义与性质](./旋转矩阵的定义与性质.md) | 旋转矩阵的定义与性质 |
|L1| ✅ | 🧊 位置编码 | [旋转矩阵的相位相消证明](./旋转矩阵的相位相消证明.md) | 旋转矩阵的性质证明 |
|L1| ✅ | 🔥 自注意力机制 | [encoder编码器](./encoder编码器.md)  | 什么是encoder编码器，encoder编码器的原理是什么 | 🔥[encoder_layer.py](exercise/encoder_layer.py)|
|L1| ✅ | 🧊 自注意力机制 | [decoder解码器](./decoder解码器.md)  | 什么是decoder解码器，decoder解码器的原理是什么 | 🔥[decoder_layer.py](exercise/decoder_layer.py)|



# 大纲
encoder
- decoder的输入
    - tokenizer嵌入层
    - 位置编码
- 前向传输


-


decoder解码器与encoder编码器的区别？


class TransformerEncoder(nn.Module):
    



Class Transformer(nn.Module):
    forward(src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.pos_embedding(src_pos)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(tgt_pos)


        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
        return dec_out