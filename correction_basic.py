# @Author:sunshine
# @Time  : 2020/5/11 上午9:12

"""
利用语言模型提供的置信度+字音和字形生成的相似度,加权输出一个分数,选取分数最高的候选字符
"""

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
from tools.char_sim import CharFuncs

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

model.summary()
C = CharFuncs('data/char_meta.txt')


def text_correction(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    probs = model.predict([[token_ids], [segment_ids]])[0][1:-1]
    topk_probs_index = np.argsort(-probs, axis=1)[:, :topk]
    true_chars = ''
    for candidate_probs, candidate_probs_index, char in zip(probs, topk_probs_index, tokens[1:-1]):
        candidate = tokenizer.decode(candidate_probs_index)
        if candidate[0] != char:
            scores = []
            candidate_prob_topk = candidate_probs[candidate_probs_index]
            for c, b in zip(candidate, candidate_prob_topk):
                sim = C.similarity(char, c, weights=(0.8, 0.2, 0.0))
                score = 0.6 * b + 0.4 * sim
                scores.append((score, char))
            if scores:
                sort_score = sorted(scores, key=lambda x: x[0], reverse=True)
                true_chars += sort_score[0][1]
            else:
                true_chars += char
        else:
            true_chars += char
    return true_chars


if __name__ == '__main__':
    text = '专家公步虎门大桥涡振原因'
    result = text_correction(text)
    print(result)
    # print(len(text), len(result))
