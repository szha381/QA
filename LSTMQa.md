
# 基于LSTM的地理中文问答


```python
# coding=utf-8
import mxnet as mx
import mxnet.ndarray as nd


class LSTMQa(object):
    """基于lstm的中文地理知识库问答
    实现
    """
    def __init__(self, batch_size, embedding, embedding_size, rnn_size, margin, std_scale, input_dim,
                 hidden_dim, output_dim, state_h, state_c, input_ques_iter, positive_ans_iter, negative_ans_iter, ctx):
        self.batch_size = batch_size
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.margin = margin
        self.params = self.get_params(std_scale, input_dim, hidden_dim, output_dim, ctx)

        # create LSTM net
        self.ques_out, self.state_h, self.state_c = self.lstm(input_ques_iter, state_h, state_c, *self.params)
        self.positive_ans_out, self.state_h, self.state_c = self.lstm(positive_ans_iter, state_h, state_c, *self.params)
        self.negative_ans_out, self.state_h, self.state_c = self.lstm(negative_ans_iter, state_h, state_c, *self.params)

        #pooling

        # 计算loss
        positive_ans_similarity = nd.cos(self.ques_out, self.positive_ans_out)
        negative_ans_similarity = nd.cos(self.ques_out, self.negative_ans_out)
        self.loss = nd.max(0, margin - positive_ans_similarity + negative_ans_similarity)


    @staticmethod
    def lstm(inputs, state_h, state_c, *params):
        """
        :param inputs 维度为batch_size * vacab_size
        :param state_h 维度为batch_size * hidden_dim
        :param state_c 维度为batch_size * vacab_size
        """
        try:
            x_inputs = nd.transpose(inputs, [1, 0, 2])
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hy, b_y] = params
        H = state_h
        C = state_c
        outputs = []
        for X in inputs:
            I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
            F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
            O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
            C_candi = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
            C = F * C + I * C_candi
            H = O * nd.tanh(C)
            Y = nd.dot(H, W_hy) + b_y
            outputs.append(Y)
        return (outputs, H, C)

    @staticmethod
    def get_params(std_scale, input_dim, hidden_dim, output_dim, ctx):
        """初始化所有参数
        """
        # 输入门参数
        W_xi = nd.random_normal(scale=std_scale, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hi = nd.random_normal(scale=std_scale, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_i = nd.zeros(hidden_dim, ctx=ctx)

        # 遗忘门参数
        W_xf = nd.random_normal(scale=std_scale, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hf = nd.random_normal(scale=std_scale, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_f = nd.zeros(hidden_dim, ctx=ctx)

        # 输出门参数
        W_xo = nd.random_normal(scale=std_scale, shape=(input_dim, hidden_dim), ctx=ctx)
        W_ho = nd.random_normal(scale=std_scale, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_o = nd.zeros(hidden_dim, ctx=ctx)

        # 候选细胞参数
        W_xc = nd.random_normal(scale=std_scale, shape=(input_dim, hidden_dim), ctx=ctx)
        W_hc = nd.random_normal(scale=std_scale, shape=(hidden_dim, hidden_dim), ctx=ctx)
        b_c = nd.zeros(hidden_dim, ctx=ctx)

        # 输出层
        W_hy = nd.random_normal(scale=std_scale, shape=(hidden_dim, output_dim), ctx=ctx)
        b_y = nd.zeros(output_dim, ctx=ctx)

        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                  b_c, W_hy, b_y]
        for param in params:
            param.attach_grad()
        return params





```
