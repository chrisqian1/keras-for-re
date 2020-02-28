import keras
import keras_bert
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.base_layer import Layer
from keras.initializers import Constant
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_attention import SelfAttention
import keras.backend as K
import keras_xlnet 
from keras_xlnet import PretrainedList, get_pretrained_paths, ATTENTION_TYPE_BI

max_seq_len = 128


class Piecewise_Maxpooling(Layer):
    def __init__(self, filter_num, seq_len, **kwargs):
        self.filter_num = filter_num
        self.seq_len = seq_len
        super(Piecewise_Maxpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Piecewise_Maxpooling, self).build(input_shape)

    def max_pool_piece1(self, x):
        conv_output = x[0]
        e1 = K.cast(x[1], 'int32')
        piece = K.slice(conv_output, [0, 0], [e1, self.filter_num])
        return K.max(piece, 0)

    def max_pool_piece2(self, x):
        conv_output = x[0]
        e1 = K.cast(x[1], 'int32')
        e2 = K.cast(x[2], 'int32')
        piece = K.slice(conv_output, [e1, 0], [e2 - e1, self.filter_num])
        return K.max(piece, 0)

    def max_pool_piece3(self, x):
        conv_output = x[0]
        e2 = K.cast(x[1], 'int32')
        piece = K.slice(conv_output, [e2, 0], [self.seq_len - e2, self.filter_num])
        return K.max(piece, 0)

    def call(self, inputs):
        assert (len(inputs) == 2)
        # ourpur of conv layer
        conv_output = inputs[0]
        # entity1 end position
        e1 = inputs[1][:, 0]
        # entity2 end position
        e2 = inputs[1][:, 1]
        # split the result into three parts
        conv_piece1 = K.map_fn(self.max_pool_piece1, (conv_output, e1), dtype='float32')
        conv_piece2 = K.map_fn(self.max_pool_piece2, (conv_output, e1, e2), dtype='float32')
        conv_piece3 = K.map_fn(self.max_pool_piece3, (conv_output, e2), dtype='float32')
        # return concatenated vector
        return K.concatenate([conv_piece1, conv_piece2, conv_piece3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2] * 3)


class Extract(Layer):
    def __init__(self, **kwargs):
        super(Extract, self).__init__(**kwargs)

    def extract_entity(self, x):
        sent = x[0]
        spos = K.cast(x[1], 'int32')
        epos = K.cast(x[2], 'int32')
        entity = K.slice(sent, [spos, 0], [epos - spos, 768])
        return K.mean(entity, 0)

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x):
        assert (len(x) == 2)
        sent = x[0]
        e1spos = x[1][:, 0]
        e1epos = x[1][:, 1]
        e2spos = x[1][:, 2]
        e2epos = x[1][:, 3]
        # extract vector for entity1 and entity2
        h1 = K.map_fn(self.extract_entity, (sent, e1spos, e1epos), dtype='float32')
        h2 = K.map_fn(self.extract_entity, (sent, e2spos, e2epos), dtype='float32')
        return [h1, h2]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][2]), (input_shape[0][0], input_shape[0][2])]


def Squeeze(x):
    return keras.backend.squeeze(x, axis=1)


def file_line2list(sfilename, fcoding='utf8', lowerID=False, stripID=True, verbose=0):
    finput = open(sfilename, 'r', encoding=fcoding)
    lines = finput.readlines()
    for i in range(len(lines)):
        if stripID: lines[i] = lines[i].strip()
        else: lines[i] = lines[i].strip('\r\n')
        if lowerID: lines[i] = lines[i].lower()
    finput.close()
    if verbose:  print('\nLoading {:,} lines from {} ...'.format(len(lines), sfilename))
    return lines


def load_pretrained_embedding_from_file(embed_file, word_dict, EMBED_DIM=100):
    """
    load pretrained embeddingmatrix
    """
    # load glove embedding
    lines = file_line2array(embed_file, sepch=' ', verbose=True)
    # filter the lines containing words in word_dict
    wlines = [values for values in lines if values[0] in word_dict]
    np.random.seed(1234)
    embed_matrix = np.random.normal(0, 1, (len(word_dict), EMBED_DIM))
    # add embedding weights to matrix
    for values in wlines:
        idx = word_dict[values[0]]
        embed_matrix[idx] = np.asarray(values[1:], dtype='float32')
    return embed_matrix, EMBED_DIM


def load_word_voc_file(filename=None, verbose=0):
    if not os.path.exists(filename): return None
    words = file_line2list(filename, verbose=verbose)
    word_dict = {word:i for i, word in enumerate(words)}
    return word_dict


def load_bert_tokenizer(bert_model_path):
    # load tokenizer and token dict from bert path
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token_dict = load_word_voc_file(dict_path, verbose=True)
    tokenizer = keras_bert.Tokenizer(token_dict)
    return token_dict, tokenizer


def load_bert_model(bert_path, verbose=0):
    # load bert model from bert path
    if verbose:
        print('\nLoading the BERT model from {} ...'.format(bert_path))
    config_path = os.path.join(bert_path, 'bert_config.json')
    checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_model = keras_bert.load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    return bert_model


def load_xlnet_tokenizer():
    # load tokenizer
    paths = get_pretrained_paths(PretrainedList.en_cased_base)
    tokenizer = keras_xlnet.Tokenizer(paths.vocab)
    return tokenizer


def create_lstm_classification_model(level, modelname, vocab_size, POS_size, num_classes, EMBED_MATRIX, embed_dim):
    """
    create lstm model
    level: sent, token
    """

    # model input
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None, None))

    # embedding layer
    word_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=Constant(EMBED_MATRIX),
                           input_length=None, mask_zero=True)(x1_in)
    if 'POS' in modelname:
        POS_embed = Embedding(input_dim=POS_size, output_dim=20, input_length=None)(x2_in)
        word_embed = Concatenate()([word_embed, POS_embed])
    # dropout layer
    word_embed = Dropout(0.2)(word_embed)
    # bilstm layer
    return_sequence = True if level == 'token' or 'Att' in modelname else False
    x = Bidirectional(LSTM(units=100, return_sequences=return_sequence, recurrent_dropout=0.1))(word_embed)

    # crf and dense layer accroding to crf and level
    if 'Crf' in modelname:
        if level == 'token':
            x = TimeDistributed(Dense(50, activation="relu"))(x)  # a dense layer as suggested by neuralNer
            output = CRF(num_classes)(x)
        else:
            x = Dense(50, activation='relu')(x)
            output = CRF(num_classes)(x)
    else:
        if level == 'token':
            output = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
        else:
            if 'Att' in modelname:
                x = SelfAttention()(x)
            output = Dense(num_classes, activation='softmax')(x)

    # build and compile model
    model = Model([x1_in, x2_in, x3_in], output)
    if 'Crf' in modelname:
        model.compile(loss=crf_loss, optimizer='rmsprop', metrics=[crf_viterbi_accuracy])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.summary()
    return model


def create_cnn_classification_model(level, model_name, vocab_size, num_classes, EMBED_MATRIX, embed_dim):
    """
    create cnn model
    level: sent, token
    """

    # model input
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(2,))

    # embedding layer
    word_embed = Embedding(input_dim=vocab_size, output_dim=100, embeddings_initializer=Constant(EMBED_MATRIX),
                           input_length=None)(x1_in)

    # convolutional layer
    model = Conv1D(filters=100, kernel_size=3)(word_embed)

    # maxpooling layer
    if level == 'sent':
        if 'Piece' in model_name:
            model = Piecewise_Maxpooling(100, 126)([model, x2_in])
        else:
            model = MaxPooling1D(pool_size=126)(model)
            model = Lambda(Squeeze)(model)
    # dense layer
    output = Dense(num_classes, activation='softmax')(model)

    # build and compile model
    model = Model([x1_in, x2_in], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    return model


def create_bert_classification_model(bert_path, level, modelname, num_classes):
    """
    create bert model
    level: sent, token
    """

    # model input
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(4,))

    # bert model layer
    bert_model = load_bert_model(bert_path)
    for l in bert_model.layers: l.trainable = True
    x = bert_model([x1_in, x2_in])

    # crf and dense layer accroding to crf and level
    if level == 'token':
        if 'Crf' in modelname:
            x = TimeDistributed(Dense(50, activation="relu"))(x)
            output = CRF(num_classes)(x)
        else:
            output = TimeDistributed(Dense(num_classes, activation="softmax"))(x)
    else:
        h0 = Lambda(lambda x: x[:, 0])(x)
        if 'Crf' in modelname:
            h0 = Dense(50, activation='relu')(h0)
            output = CRF(num_classes)(h0)
        else:
            # relation bert
            if 'R_Bert' in modelname:
                [h1, h2] = Extract()([x, x3_in])
                h0 = Dense(100, activation = "relu")(h0)
                h1 = Dense(100, activation = "relu")(h1)
                h2 = Dense(100, activation = "relu")(h2)
                h = Concatenate()([h0, h1, h2])
                output = Dense(num_classes, activation='softmax')(h)
            else:
                output = Dense(num_classes, activation='softmax')(h0)

    # build and compile model
    model = Model([x1_in, x2_in, x3_in], output)
    if 'Crf' in modelname:
        model.compile(loss=crf_loss, optimizer=Adam(1e-5), metrics=[crf_viterbi_accuracy])
    else:  
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['categorical_accuracy'])
    model.summary()
    return model


def create_xlnet_classification_model(batch_size, level, num_classes):
    """
    create bert model
    level: sent, token
    """

    # load xlnet model
    paths = get_pretrained_paths(PretrainedList.en_cased_base)
    model = keras_xlnet.load_trained_model_from_checkpoint(config_path=paths.config, checkpoint_path=paths.model, batch_size=batch_size, memory_len=0, target_len=max_seq_len, in_train_phase=False, attention_type=ATTENTION_TYPE_BI)

    # crf and dense layer accroding to crf and level
    if level == 'token':
        output = TimeDistributed(Dense(num_classes, activation="softmax"))(model.output)
    else:
        x = Lambda(lambda x: x[:, 0])(model.output)
        output = Dense(num_classes, activation='softmax')(x)

    # build and compile model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(optimizer=Adam(lr=3e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    return model
