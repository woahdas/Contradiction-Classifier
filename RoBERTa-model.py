#this is using the RoBERTa model, which is made by Facebook.
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from transformers import TFAutoModel, AutoConfig
from tensorflow.keras.layers import Dense, Input, Dropout
from transformers import BertTokenizer


# TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()  # for CPU and single GPU
    print('No. of replicas:', strategy.num_replicas_in_sync)

# getting the data
# 0 is entailment, 1 is neutral, 2 is contradiction

train = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")

# getting the RoBERTa model
model_name = 'jplu/tf-xlm-roberta-large'
tokenizer = BertTokenizer.from_pretrained(model_name)

def get_xlm_roberta(modelname=model_name):
    conf = AutoConfig.from_pretrained(modelname)
    conf.output_hidden_states = True
    model = TFAutoModel.from_pretrained(modelname, config=conf)
    return model

#building the model
def build_model(xlm_roberta, max_len=256, p=0.5):

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    _, _, hidden_states = xlm_roberta([input_ids, attention_mask])
    x = tf.concat(hidden_states[-2:], -1)
    x = tf.concat((tf.reduce_mean(x, 1), tf.reduce_max(x, 1)), -1)
    x = Dropout(rate=0.5)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=(input_ids, attention_mask), outputs=out)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1, name='binary_crossentropy')

    model.compile(Adam(lr=1e-5), loss=loss_fn, metrics=[tf.metrics.AUC()])

    return model

with strategy.scope():
    xlm_roberta = get_xlm_roberta()
    model = build_model(xlm_roberta)
model.summary()



# tokenizing the data
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(hypotheses, premises, tokenizer):
    num_examples = len(hypotheses)

    sentence1 = tf.ragged.constant([
        encode_sentence(s)
        for s in np.array(hypotheses)])
    sentence2 = tf.ragged.constant([
        encode_sentence(s)
        for s in np.array(premises)])

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


train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)

# creating and training the model
max_len = 50


model.fit(train_input, train.label.values, epochs=2, verbose=1, batch_size=64, validation_split=0.2)

test = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)

# make and submit the predictions
predictions = [np.argmax(i) for i in model.predict(test_input)]
submission = test.id.copy().to_frame()
submission['prediction'] = predictions
submission.to_csv("submission1.csv", index=False)
