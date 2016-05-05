from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler

def get_rate(epoch):
    if epoch < 3:
        return .01
    if epoch < 6:
        return .005
    return .002


def get_conv_rnn(embedding_weights, **options):
    """
    Required parameters :
        max_features (default 150001)
        maxlen (default 101)
        embedding_size (default 200)
        filter_length (default 4)
        nb_filter (default 32)
        pool_length (default 3)
        gru_output_size (default 100)
        gru_dropout_w 
    """

    # Embedding
    max_features = options.get('max_features', 19855)
    maxlen = options.get('maxlen', 101)
    embedding_size = options.get('embedding_size', 200)
    embedding_dropout = options.get('embedding_dropout', .45)

    # Convolution
    filter_length = options.get('filter_length', 4)
    nb_filter = options.get('nb_filter', 64)
    pool_length = options.get('pool_length', 3)

    # gru
    gru_output_size = options.get('gru_output_size', 100)
    gru_dropout = options.get('gru_dropout', .5)
    gru_l2_coef_w = options.get('gru_l2_coef_w', .0001)
    gru_l2_coef_u = options.get('gru_l2_coef_u', .0001)

    # learning
    clipnorm = options.get('clipnorm', 5.)

    print('Build model...')

    model = Sequential()
    if embedding_weights is not None:
        model.add(Embedding(max_features, embedding_size, input_length=maxlen, weights=[embedding_weights]))
    else:
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(embedding_dropout))
    # model.add(Convolution1D(nb_filter=nb_filter,
    #                       filter_length=filter_length,
    #                       border_mode='valid',
    #                       activation='relu',
    #                       subsample_length=1))
    # model.add(MaxPooling1D(pool_length=pool_length))
    model.add(GRU(gru_output_size, dropout_W=gru_dropout, dropout_U=gru_dropout,
                  W_regularizer=l2(gru_l2_coef_w), U_regularizer=l2(gru_l2_coef_u)))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer=Adam(clipnorm=clipnorm),
                metrics=['accuracy'])
    return model

def train(model, X_train, y_train, X_test, y_test, **options):
    """
    Kwarg options:
        batch_size (default 32)
        nb_epoch (default 10)
    """
    # Training
    batch_size = options.get('batch_size', 32)
    nb_epoch = options.get('nb_epoch', 10)

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_data=(X_test, y_test), callbacks=[LearningRateScheduler(get_rate)])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
