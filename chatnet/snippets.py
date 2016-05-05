from chatnet import gather, prep, keras_model
import pandas as pd
import numpy as np
from functools import partial
from matplotlib.pylab import plt

# useful looking snippets

# test_calibration_df['session'] = test_calibration_df['chunk_ids'].map(lambda v: v.split('_')[0])
# test_calibration_df['chunk_ix'] = test_calibration_df['chunk_ids'].map(lambda v: v.split('_')[1])


def calibration_plot(prob, ytest):
    # stolen from stackoverflow!
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))

    #group outcomes into bins of similar probability
    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]
    
    #freshness ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
        
    #the calibration plot
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
    plt.ylabel("Empirical P(Product)")
    
    #the distribution of P(fresh)
    ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)
    
    plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
            width=.95 * (bins[1] - bins[0]),
            fc=p[0].get_color())
    
    plt.xlabel("Predicted P(Product)")
    plt.ylabel("Number")


def is_middling_group(df):
    return df.count() == df[df['predicted'].map(lambda v: v > .2 and v < .8)].count()





class Pipeline(object):
    """That's right I'm re-implementing Pipeline"""
    def __init__(self, vocab_size, value_filter=None,
                 data_col=None, id_col=None, label_col=None, features=[],
                 binary=True, prepend_sender=True, binary_options={u'Product', u'Service'},
                 keras_model_options=None):
        self.vocab_size=vocab_size
        self.value_filter = value_filter or (lambda v: True)
        self.data_col = data_col or 'msgs'
        self.id_col = id_col or 'Chat Session ID'
        self.label_col = label_col or 'Chat Type'
        self.features = features
        self.prepend_sender = prepend_sender
        self.label_filter = lambda v: True if not binary else v in binary_options
        self.keras_model_options = keras_model_options or {}
 
    def setup(self, df, **to_matrices_kwargs):
        # 

        to_matrices_kwargs.setdefault('seed', 212)
        to_matrices_kwargs.setdefault('test_split', .18)
        to_matrices_kwargs.setdefault('chunk_size', 100)

        cols = gather.get_message_cols(df)

        mapper = partial(gather.assemble_messages, message_cols=cols,
                         value_filter=self.value_filter, prepend_sender=self.prepend_sender)
        
        binary_df = pd.DataFrame(df[df[self.label_col].map(self.label_filter)])

        print("Aggregating messages...")
        binary_df[self.data_col] = binary_df.apply(mapper, axis=1)
        
        tp = prep.TextPrepper()

        data = pd.DataFrame(binary_df[
            [self.data_col, self.id_col, self.label_col] + self.features
            ])
        
        print("Counting words...")
        word_counts = prep.get_word_counts(data[self.data_col], tp)
        
        print("Finding embedding...")
        nonembeddable = prep.get_nonembeddable_set(word_counts)
        
        word_index = prep.get_word_index(word_counts, nb_words=self.vocab_size, nonembeddable=nonembeddable)

        embedding_weights, n_symbols = prep.get_embedding_weights(word_index)

        print("Making numeric sequences...")
        self.learning_data = (X_train, y_train, train_ids), (X_test, y_test, test_ids) = \
            tp.to_matrices(data, word_index, **to_matrices_kwargs)
        
        self.model = keras_model.get_conv_rnn(embedding_weights, max_features=n_symbols, **self.keras_model_options)

    def run(self, **training_options):
        (X_train, y_train, train_ids), (X_test, y_test, test_ids) = self.learning_data
        keras_model.train(self.model, X_train, y_train, X_test, y_test, **training_options)





# def set_learning_rate(hist, learning_rate = 0, activate_halving_learning_rate = False, new_loss =0, past_loss = 0, counter = 0, save_model_dir=''):
#     if activate_halving_learning_rate and (learning_rate>=0.0001):
#         if counter == 0:
#             new_loss = hist.history['loss'][0]
#             if new_loss>=(past_loss): #you want at least a 0.5% loss decrease compared to the previous iteration
#                 learning_rate = float(learning_rate)/float(2)
#                 print 'you readjusted the learning rate to', learning_rate
#                 with open('models/'+save_model_dir+'/'+'history_report.txt', 'a+') as history_file:
#                     history_file.write('For next Iteration, Learning Rate has been reduced to '+str(learning_rate)+'\n\n')
#                     with open('history_reports/'+save_model_dir+'_'+'history_report.txt', 'a+') as history_file:
#                         history_file.write('For next Iteration, Learning Rate has been reduced to '+str(learning_rate)+'\n\n')

#             past_loss = new_loss
#         return (learning_rate, new_loss, past_loss)




# class CategoricalTransformer(TransformerMixin):

#     def fit(self, X, y=None, *args, **kwargs):
#         self.columns_ = X.columns
#         self.cat_columns_ = X.select_dtypes(include=['category']).columns
#         self.non_cat_columns_ = X.columns.drop(self.cat_columns_)

#         self.cat_map_ = {col: X[col].cat.categories
#                          for col in self.cat_columns_}
#         self.ordered_ = {col: X[col].cat.ordered
#                          for col in self.cat_columns_}

#         self.dummy_columns_ = {col: ["_".join([col, v])
#                                      for v in self.cat_map_[col]]
#                                for col in self.cat_columns_}
#         self.transformed_columns_ = pd.Index(
#             self.non_cat_columns_.tolist() +
#             list(chain.from_iterable(self.dummy_columns_[k]
#                                      for k in self.cat_columns_))
#         )

#     def transform(self, X, y=None, *args, **kwargs):
#         return (pd.get_dummies(X)
#                   .reindex(columns=self.transformed_columns_)
#                   .fillna(0))