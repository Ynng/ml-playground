import os
import tensorflow
import numpy as np
import torch
import re
import jieba
import random


def load_data():
    xs = []
    ys = []
    with open(os.path.dirname(os.path.abspath(__file__))+'/online_shopping_10_cats.csv', 'r', encoding='utf-8') as f:
        line = f.readline()  # escape first line"label review"
        while line:
            line = f.readline()
            if not line:
                break
            contents = line.split(',')

            # if contents[0]=="书籍":
            # 	continue

            label = int(contents[1])
            review = contents[2]
            if len(review) > 20:
                continue

            xs.append(review)
            ys.append(label)

    xs = np.array(xs)
    ys = np.array(ys, dtype='float32')

    # 打乱数据集
    indies = [i for i in range(len(xs))]
    random.seed(666)
    random.shuffle(indies)
    xs = xs[indies]
    ys = ys[indies]

    m = len(xs)
    cutpoint = int(m*4/5)
    x_train = xs[:cutpoint]
    y_train = ys[:cutpoint]

    x_test = xs[cutpoint:]
    y_test = ys[cutpoint:]

    print('Sample size:%d' % (len(xs)))
    print('Train set size:%d' % (len(x_train)))
    print('Test set size:%d' % (len(x_test)))

    return x_train, y_train, x_test, y_test


def createWordIndex(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis=0)
    word_dic = {}
    voca = []
    for sentence in x_all:
        sentence = re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)
        cut = jieba.cut(sentence)
        for word in cut:
            word_dic[word] = word_dic.get(word, 0) + 1
            voca.append(word)
    word_dic = sorted(word_dic.items(), key=lambda kv: kv[1], reverse=True)
    word_to_ix = {word[0]: i for i, word in enumerate(word_dic)}

    print(f"voca: {len(word_dic)}")
    return len(word_dic), word_to_ix


def word2Index(words, word_index):
    vecs = []
    max_len = 0
    for sentence in words:
        sentence = re.sub(
            "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", sentence)
        cut = jieba.cut(sentence)
        index = []

        index = [float(word_index[word]) for word in cut]
        max_len = max(max_len, len(index))
        vecs.append(np.array(index, dtype='int64'))
        
    vecs = [np.pad(vec, (0, max_len - len(vec)), 'constant') for vec in vecs]

    return np.array(vecs)
