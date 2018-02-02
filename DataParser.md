
# 数据处理


```python
# coding=utf-8
from Collections import defaultdict

"""Training set：指定一个Question只有一个正确答案，同时对应20个错误答案
为平衡训练，暂采取 正例：反例 = 1 ：1进行训练（即讲正确答案复制19次）
   Testing set：一个Question给出20个候选答案，其中可以给定多个正确答案
"""


def load_embedding(embedding_file, embedding_dim):
    """加载预训练好的embedding，一般是txt文件
    :param embedding_file 训练好的embedding文本文件
    :param embedding_dim embedding维度
    :returns embedding矩阵，词汇表字典((key,value)= (vocabulary, index))
    """
    embedding_matrix = []
    word2vocabulary_index = defaultdict(list)
    word2vocabulary_index['UNKNOWN'] = 0  # 未登录词
    embedding_matrix.append([0] * embedding_dim) # 设置未登录词的embedding

    with open(embedding_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_array = line.split(" ")  # embedding以空格分隔，如the 0.232 0.87323.....
            word2vocabulary_index[line_array[0]] = len(word2vocabulary_index) # 从1开始，0为未登录词
            embedding_matrix.append[[float(val) for val in line_array[1:]]]
    return embedding_matrix, word2vocabulary_index


def sentence2embedding_index(sentence, word2vocabulary_index, max_len):
    """将分好词的句子中映射为词典索引列表(亦为embedding索引列表)
       此处会处理UNKNOWN词，或者句子可以忽视数字、特殊符号等
    :param sentence 分好词的句子（词list）
    :param word2vocabulary_index 词典
    :param max_len 句子最大长度，大于max_len则后面词语丢弃（此处采用定长表示）
    :return sentence在词典索引列表
    """
    index = []
    i = 0
    unknow_index = word2vocabulary_index.get('UNKNOWN')
    for word in sentence:
        if word in word2vocabulary_index:
            index[i] = word2vocabulary_index[word]
        else:
            index[i] = unknow_index   # 此处未登录词全用一个相同的embedding表示
        i += 1
        if i >= max_len:
            break
    return index


def load_training_data(file_path, word2embedding_index, sentence_max_len):
    """加载训练集
        训练集行格式：question \t answer \t label(1或0)
    :param word2embedding_index 词典
    :param file_path 文件路径
    :param sentence_max_len 问题或答案最大长度、该长度后的词语丢弃（本实验设置等长）
    :return questions、answers、labels、question_ids(分配的问题id，id有重复)
    """
    question, question_id = "", 0
    questions, answers, labels, question_ids = [], [], [], []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_array = line.split("\t")
            if len(line_array) != 3:   # 格式不正确
                continue
            if line_array[0] != question:
                question = line_array[0]
                question_id += 1
            ques_embedding_idx = sentence2embedding_index(line_array[0], word2embedding_index, sentence_max_len)
            ans_embedding_idx = sentence2embedding_index(line_array[1], word2embedding_index, sentence_max_len)
            label = int(line_array[3])

            questions.append(ques_embedding_idx)
            answers.append(ans_embedding_idx)
            labels.append(label)
            question_ids.append(question_id)
        return questions, answers, labels, question_ids


def data_iter(questions, answers, labels, question_ids, batch_size):
    """batch迭代器， (问题，正解，错解)即(q, a+, a-)
    """
    sample_len = len(question_ids)
    batch_num = sample_len / batch_size + 1
    sample_i = 0
    question_iter, positive_answer_iter, negative_answer_iter = [], [], []
    for batch_i in range(batch_num):
        # per batch,
        for ques_i in range(batch_i * batch_size, min((batch_i+1) * batch_size, sample_len):
        # 每个问题寻找其正例、反例
            negative_ans_count = 0
        while question_ids[sample_i] == ques_i:
            if labels[sample_i] = 0:
                negative_answer_iter.append(answers[sample_i])
                question_iter.append(questions[sample_i])
                negative_ans_count += 1
            else:
                positive_answer = answers[sample_i]
            sample_i += 1
            # 此处复制正例使与反例达到1：1
        positive_answer_iter.extend([positive_answer] * negative_ans_count)
        yield nd.array(question_iter), nd.array(positive_answer_iter), nd.array(negative_answer_iter)







```
