from keras.datasets import imdb
import numpy as np


def load_online():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
    return (x_train, y_train), (x_test, y_test)

def load_disk(fileName = "./imdb.npz"):
    data = np.load(fileName)


    x_test = data['x_test']
    x_train = data['x_train']
    y_test = data['y_test']
    y_train = data['y_train']

    return (x_train, y_train), (x_test, y_test)


def build_vocab(x_train, x_test):
    vocab_dict = dict()
    lenList = [len(l) for l in x_train + x_test]

    print(max(lenList))
    print(min(lenList))
    print(sum(lenList))
    mean = sum(lenList)/float(len(lenList))

    print(mean)
    for l in x_train + x_test:
        for d in l:
            if d not in vocab_dict: 
                vocab_dict[d] = 1
            else:
                vocab_dict[d] += 1

    voc_dim = max(vocab_dict.keys()) + 1
    return voc_dim

def proprocess(x_train, x_test, max_len = 1000):
    from keras.preprocessing.sequence import pad_sequences
    x_train = pad_sequences(x_train, padding = 'post', maxlen = max_len)
    x_test = pad_sequences(x_test, padding = 'post', maxlen = max_len)

    return x_train, x_test;
    
def build_model(voc_dim, max_len, embed_dim):
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Conv2D, Reshape, Conv1D
    from keras.layers import MaxPool2D, PReLU, AvgPool2D, MaxPooling1D
    from keras.layers.core import Flatten, Dense, Dropout, Lambda
    from keras.optimizers import Adam

    desc_inp = Input(shape=(max_len,), name="test")
    emb_desc = Embedding(voc_dim, embed_dim)(desc_inp)
    emb_desc = SpatialDropout1D(.4)(emb_desc)
    emb_desc = Reshape((max_len, embed_dim, 1))(emb_desc)


    filter_sizes=[1,3]
    convs = []
    for filter_size in filter_sizes:
        conv = Conv2D(32, kernel_size=(filter_size,embed_dim), 
                        kernel_initializer="normal", activation="relu")(emb_desc)
        convs.append(MaxPool2D(pool_size=(max_len-filter_size+1,1))(conv))
        

    embs = Flatten()(concatenate(convs))
    embs = Dropout(0.2)(Dense(64, activation="relu", kernel_initializer="he_normal")(embs))

    x = BatchNormalization()(embs)
    
    dense_n = [256, 64]
    for n in dense_n:
        x = BatchNormalization()(x)
        x = Dense(n, activation="relu", kernel_initializer="he_normal")(x)

    x = BatchNormalization()(x)
    x = Dropout(.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=desc_inp, outputs=out)
    opt = Adam()
    model.compile(optimizer=opt, loss= 'binary_crossentropy')
    return model



def model_train(model, x_train, y_train):
    from sklearn.metrics import accuracy_score, recall_score, auc
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    skf = StratifiedKFold(n_splits = 5)
    for train_index, test_index in skf.split(x_train, y_train):
        train, test = x_train[train_index], x_train[test_index]
        train_target,  test_target = y_train[train_index], y_train[test_index]
        model.fit(train, train_target, validation_split = 0.2, epochs=10, batch_size=128)
        pred = model.predict(test)
        print("eval ---")
        print('accuracy', accuracy_score(pred, test_target))
        print('recall', recall_score(pred, test_target))
        print('auc',auc(pred, test_target))
    
    
max_len =1000
embed_dim = 10
(x_train, y_train), (x_test, y_test) = load_disk()
voc_dim = build_vocab(x_train, x_test)
x_train, x_test = proprocess(x_train, x_test)
model = build_model(voc_dim, max_len, embed_dim)
model_train(model, x_train, y_train)




