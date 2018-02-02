
# 问答训练


```python
# coding=utf-8
import mxnet as mx
from mxnet.gluon import ndarray as nd
from mxnet import autograd
import DataParser as data_parser
from LSTM_QA import LSTMQa
import utils


if __name__ = '__main__':
    embedding_file = "data\\wiki_zhs_embedding.model.txt"
    embedding_dim = 300
    training_set_file = "data\\training_set"
    sentence_max_len = 50

    batch_size = 20
    input_dim = 50
    hidden_dim = 200
    output_dim = 50
    std_scale = .01
    margin = 0.3
    learning_rate = 0.3
    epochs = 20

    # 加载embedding
    embedding, word2Embedding_index = data_parser.load_embedding(embedding_file)

    # 加载训练集
    questions, answers, labels, question_ids = data_parser.load_training_data(training_set_file,
                                                                              word2Embedding_index, sentence_max_len)

    ctx = utils.try_gpu()
    # 训练模型
    for e in range(epochs):
        training_err = 0.0
        training_acc = 0.0
        state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        for question, positive_ans, negative_ans in data_parser.data_iter(questions, answers, labels, question_ids, batch_size):
                with autograd.ward():
                    lstm = LSTMQa(batch_size, embedding, embedding_dim, hidden_dim, margin, input_dim, hidden_dim,
                            output_dim, state_h, state_c, question, positive_ans, negative_ans, ctx)
                lstm.loss.backward()
                utils.SGD(lstm.params, learning_rate)
                training_acc += nd.sum(lstm.loss).asscalar()

```
