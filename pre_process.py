# _*_ coding:utf-8 _*_
# Time   :  上午11:32
# Author : sunshine
from bs4 import BeautifulSoup
import codecs
from langconv import Converter
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
            tag = tmp[_id]
            tag['correction_text'] = tag['correction_text'].replace(wrong, correction)
            tag['num'] = tag['num'] + 1

        for k, v in tmp.items():
            wt.write(str(v['num']) + '\t' + conv.convert(v['text']) + '\t' + conv.convert(v['correction_text']) + '\n')
    print(len(texts))


if __name__ == '__main__':
    file = 'sighan7csc_release1.0/SampleSet/1.txt'
    target = 'sigha7csc.txt'
    parse_data(file, target)
