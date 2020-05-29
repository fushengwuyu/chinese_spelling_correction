# @Author:sunshine
# @Time  : 2020/5/12 下午1:27

import json
from bert4keras.tokenizers import load_vocab, Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.optimizers import AdaFactor
from keras.callbacks import Callback
import keras.backend as K
from tqdm import tqdm

max_len = 64
config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'

train_data = json.load(open('data/train_data.json', 'r', encoding='utf-8'))
valid_data = json.load(open('data/valid_data.json', 'r', encoding='utf-8'))

# 加载精简词表
token_dict, keep_words = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict, do_lower_case=True)


class MyDataGenerator(DataGenerator):
    def __iter__(self, random=True):
        """
        单条样本格式: [cls]错误词汇[sep][mask][mask]..[sep]
        :param random:
        :return:
        """
        batch_tokens_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):
            wrong, right = D
            # segment_ids也作为mask输入
            token_ids, segment_ids = tokenizer.encode(first_text=wrong, second_text=right, max_length=max_len * 2)

            batch_tokens_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_tokens_ids) == self.batch_size or is_end:
                batch_tokens_ids = sequence_padding(batch_tokens_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)

                yield [batch_tokens_ids, batch_segment_ids], None
                batch_tokens_ids, batch_segment_ids = [], []


# 构建模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    application='unilm',
    keep_tokens=keep_words
)

y_true = model.input[0][:, 1:]
y_mask = model.input[1][:, 1:]
y_pred = model.output[:, :-1]

cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=AdaFactor(learning_rate=1e-3))
model.summary()


def ge_answer(wrong):
    """
    解码
    :param wrong:
    :return:
    """
    wrong_token_ids, _ = tokenizer.encode(wrong)
    token_ids = wrong_token_ids + [tokenizer._token_mask_id] * max_len + [tokenizer._token_end_id]
    segemnt_ids = [0] * len(token_ids)
    probas = model.predict([np.array([token_ids]), np.array([segemnt_ids])])[0]
    proba_ids = probas.argmax(axis=1)
    useful_index = proba_ids[np.where(proba_ids != 3)]
    if any(useful_index):
        answer = tokenizer.decode(useful_index)
    else:
        answer = tokenizer.decode(proba_ids[:len(wrong)])
    return answer


def evalute(valid_data):
    X, Y = 1e-10, 1e-10
    for item in tqdm(valid_data):
        wrong, right = item
        pred = ge_answer(wrong)
        X += pred == right
        Y += 1
    precision = X / Y
    return precision


class Evaluator(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save('models/best_seq2seq_model.h5')


if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = MyDataGenerator(train_data, batch_size=8)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
