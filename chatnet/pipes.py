from . import logger
from chatnet import gather, prep
import pandas as pd
from functools import partial

class Pipeline(object):
    """
    Transformer helper functions and state checkpoints
    to go from text data/labels to model-ready numeric data
    """
    def __init__(self, vocab_size, value_filter=None,
                 data_col=None, id_col=None, label_col=None, features=[],
                 binary=True, prepend_sender=True, binary_options={u'Product', u'Service'},
                 df=None, message_key=None):
        self.vocab_size=vocab_size
        self.value_filter = value_filter or (lambda v: True)
        self.data_col = data_col or 'msgs'
        self.id_col = id_col or 'Chat Session ID'
        self.label_col = label_col or 'Chat Type'
        self.features = features
        self.prepend_sender = prepend_sender
        self.label_filter = lambda v: True if not binary else v in binary_options
        self.word_index_kwargs = dict(nb_words=vocab_size)
        self.message_key = message_key
        if df:
            self.setup(df)
 
    def _tokenize(self, df, message_key=''):
        if not message_key:
            cols = gather.get_message_key(df)
            message_iterator = gather.get_wide_columns_message_iterator(cols)
        elif isinstance(message_key, (unicode, str)):
            message_iterator = gather.get_dense_column_message_iterator(message_key)
        elif isinstance(message_key, list):
            message_iterator = gather.get_wide_columns_message_iterator(cols)

        mapper = partial(gather.assemble_messages, message_iterator=message_iterator,
                         value_filter=self.value_filter, prepend_sender=self.prepend_sender)

        df[self.data_col] = df.apply(mapper, axis=1)


    def _set_token_data(self, df):
        label_filtered_df = pd.DataFrame(df[df[self.label_col].map(self.label_filter)])

        if self.data_col not in df.columns:
            self.tokenize(df, message_key=self.message_key)
        
        self.tp = prep.TextPrepper()

        self.data = data = pd.DataFrame(label_filtered_df[
            [self.data_col, self.id_col, self.label_col] + self.features
            ])
        
        logger.info("Counting words...")

        self.word_counts = prep.get_word_counts(data[self.data_col], self.tp)

    def _set_vocabulary(self):
        self.word_index = prep.get_word_index(self.word_counts, **self.word_index_kwargs)

    def _set_learning_data(self, **to_matrices_kwargs):
        to_matrices_kwargs.setdefault('seed', 212)
        to_matrices_kwargs.setdefault('test_split', .18)
        to_matrices_kwargs.setdefault('chunk_size', 100)
        to_matrices_kwargs.setdefault('data_col', self.data_col)
        to_matrices_kwargs.setdefault('id_col', self.id_col)
        to_matrices_kwargs.setdefault('label_col', self.label_col)
        logger.info("Making numeric sequences...")

        self.learning_data = (X_train, y_train, train_ids), (X_test, y_test, test_ids) = \
            self.tp.to_matrices(self.data, self.word_index, **to_matrices_kwargs)

    def setup(self, df):
        self._set_token_data(df)
        self._set_vocabulary()
        self._set_learning_data()
