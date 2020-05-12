# _*_ coding:utf-8 _*_
# Time   :  上午11:32
# Author : sunshine
from bs4 import BeautifulSoup
import codecs
from tools.langconv import Converter
import random
import json
conv = Converter('zh-hans')


def parse_data(file, target):
    wt = open(target, 'a', encoding='utf-8')
    origin_data = codecs.open(file, 'r').read()

    bfs = BeautifulSoup(origin_data, 'html5lib').find_all('essay')
    texts = []
    for bf in bfs:
        passages = bf.find_all('passage')
        mistakes = bf.find_all('mistake')
        tmp = {}
        for passage in passages:
            _id = passage.get('id')
            _text = passage.text
            tmp[_id] = {"text": _text, "num": 0, "correction_text": _text}
            texts.append(_text)
        for mistake in mistakes:
            _id = mistake.get('id')
            wrong = mistake.wrong.text
            correction = mistake.correction.text
            if _id not in tmp:
                continue
            tag = tmp[_id]
            tag['correction_text'] = tag['correction_text'].replace(wrong, correction)
            tag['num'] = tag['num'] + 1

        for k, v in tmp.items():
            wt.write(str(v['num']) + '\t' + conv.convert(v['text']) + '\t' + conv.convert(v['correction_text']) + '\n')
    print(len(texts))


def make2data(file, target, p=0.5):
    """
    生成训练数据, 添加语序和多字少字错误
    :return:
    """
    data = []
    with codecs.open(file, 'r', encoding='utf-8') as rd:
        for line in rd:
            """
            1	首先用嗅觉登看水果	首先用嗅觉查看水果
            """
            try:
                num, wrong, right = line.strip('\n').split('\t')
            except:
                print(line)
                continue
            p1 = random.random()
            if p1 <= p:  # 以0.2的几率打乱数据
                wrong_list = list(right)
                p2 = random.randint(0, 1)
                if p2:
                    # 交换顺序
                    index_start = random.choice(range(len(wrong_list) - 1))
                    char = wrong_list[index_start + 1]
                    wrong_list[index_start] = char
                else:
                    # 多字,少字
                    indexs = random.sample(range(len(wrong_list) - 1), k=2)
                    p3 = random.randint(0, 1)
                    if p3:
                        # 多字
                        wrong_list.insert(indexs[1], wrong_list[indexs[0]])
                    else:
                        # 少字
                        wrong_list.pop(indexs[0])
                wrong = ''.join(wrong_list)
            data.append((wrong, right))
    random.shuffle(data)
    valid_data = data[:len(data) // 10]
    train_data = data[len(data) // 10:]
    json.dump(valid_data, open('../data/valid_data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(train_data, open('../data/train_data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == '__main__':
    file = '../data/sighan8csc_release1.0/Training/SIGHAN15_CSC_B2_Training.sgml'
    target = '../data/sighan8_2.txt'
    # parse_data(file, target)

    make2data('../data/train_all.txt', '../data/train.txt')