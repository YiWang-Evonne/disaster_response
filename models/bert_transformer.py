import pandas as pd
from bert import tokenization
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow_hub
import tensorflow as tf
import numpy as np
"""
we could also use bert to encode our text. However, this is too slow to run with the resource we have.
"""


module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = tensorflow_hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2"
bert = tensorflow_hub.load(BERT_MODEL)

from tqdm import tqdm


def mini_batch(all_segments, all_tokens, all_masks, batch_size):
    """
    create mini batches for encoding.
    :param all_segments:
    :param all_tokens:
    :param all_masks:
    :param batch_size:
    :return:
    """
    for i in range(0, len(all_segments), batch_size):
        curr_batch = {'input_type_ids': tf.convert_to_tensor(all_segments[i:i + batch_size],
                                                             np.int32, name='inputs/input_word_ids'),
                      'input_word_ids': tf.convert_to_tensor(all_tokens[i:i + batch_size],
                                                             np.int32, name='inputs/input_type_ids'),
                      'input_mask': tf.convert_to_tensor(all_masks[i:i + batch_size],
                                                         np.int32, name='inputs/input_mask')}
        yield curr_batch


class BertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model, tokenizer, max_len=512, batch_size: int = 16):
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, texts):
        """
        take texts and then convert to embedding vectors.
        :param texts:
        :return:
        """
        all_tokens = []
        all_masks = []
        all_segments = []

        for text in texts:
            text = self.tokenizer.tokenize(text)

            text = text[:self.max_len - 2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = self.max_len - len(input_sequence)

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * self.max_len

            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)
        pbar_total = len(texts) // self.batch_size + 1
        pbar_desc = 'Processing USE'
        output = np.vstack([
            self.model(texts_batch)['pooled_output'].numpy() for texts_batch in tqdm(
                mini_batch(all_masks, all_segments, all_masks, self.batch_size),
                total=pbar_total,
                desc=pbar_desc
            )
        ])
        return output

