#this is using the RoBERTa model, which is made by Facebook.
#Couldn't get this to work on PyCharm, it wouldn't install pytorch or torch (either is required for RoBERTa to function).
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from transformers import RobertaConfig, RobertaModel
import pytorch #needed for RoBERTa
# TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()  # for CPU and single GPU
    print('No. of replicas:', strategy.num_replicas_in_sync)

train = pd.read_csv("Les.txt", delimiter="\t")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


configuration = RobertaConfig()

model = RobertaModel(configuration)

configuration = model.config




# tokenizing the data
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(train, tokenizer):
    num_examples = len(train)

    sentence1 = tf.ragged.constant([
        encode_sentence(s)
        for s in np.array(train)])
    sentence2 = tf.ragged.constant([
        encode_sentence(s)
        for s in np.array(train)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


train_input = bert_encode(train.values, train.values, tokenizer)

# creating and training the model
max_len = 50


model.fit(train_input, train.label.values, epochs=2, verbose=1, batch_size=64, validation_split=0.2)

predictions = model.predict()
f = open("LesGen.txt", "x")
f.close()

f = open("LesGen.txt", "a")
f.write(predictions)
f.close()
