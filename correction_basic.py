# @Author:sunshine
# @Time  : 2020/5/11 上午9:12

"""
利用语言模型提供的置信度+字音和字形生成的相似度,加权输出一个分数,选取分数最高的候选字符
"""

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
from char_sim import CharFuncs

config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'
topk = 3


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


tokenizer = OurTokenizer(vocab_path)

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)  # 建立模型，加载权重
C = CharFuncs('data/char_meta.txt')


def text_correction(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    true_sentence = ''
    for i in range(1, len(token_ids) - 1):
        token_ids[i] = tokenizer._token_dict['[MASK]']
        probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
        top_probas_index = np.argsort(-probas[i])[:topk]
        top_probas = probas[i][top_probas_index]
        candidate = tokenizer.decode(top_probas_index)
        if candidate[0] != tokens[i]:  # 最优候选字符若与原字符不一样,则启动纠错程序
            scores = []
            for char, confident in dict(zip(candidate, top_probas)).items():
                sim = C.similarity(char, tokens[i], weights=(0.8, 0.2, 0.0))
                score = 0.6 * confident + 0.4 * sim
                scores.append((score, char))
            if scores:
                sort_score = sorted(scores, key=lambda x: x[0], reverse=True)
                true_sentence += sort_score[0][1]

                # 将修改后的正确字符加入到tokens中
                token_ids[i] = tokenizer.token_to_id(sort_score[0][1])
            else:
                true_sentence += tokens[i]
                token_ids[i] = tokenizer.token_to_id(tokens[i])
        else:
            true_sentence += tokens[i]
            token_ids[i] = tokenizer.token_to_id(tokens[i])
    return true_sentence


if __name__ == '__main__':
    text = '专家公步虎门大桥涡振原因'
    result = text_correction(text)
    print(result)
    print(len(text), len(result))
