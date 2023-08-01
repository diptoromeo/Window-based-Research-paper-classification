import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nltk
import regex
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import array
import pickle
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import Bidirectional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


##===============================Nltk abstract_words tokenize======================================
with open('FGFSJournal.txt', 'rt', encoding='UTF8') as file:
    FGFS_abstracts = []
    for line in file:
        if '<abstract>' in line:
            abstract = line.split('</abstract>')[0].split('<abstract>')[-1]
            abstract = ''.join(i for i in abstract if not i.isdigit())
            abstract = regex.sub('[^\w\d\s]+', '', abstract)
            ##abstract = nltk.sent_tokenize(abstract)
            abstract = nltk.word_tokenize(abstract)
            stop_words = set(stopwords.words('english'))
            filtered_sentence_abstract = [w.lower() for w in abstract if
                                          w.lower() not in punctuation and w.lower() not in stop_words]
            tagged_list = nltk.pos_tag(filtered_sentence_abstract)
            nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
            lm = WordNetLemmatizer()
            singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
            FGFS_abstracts.append(singluar_form)


# ## ===============================pre-traind word2vec data==========================================
CBOW_embed_2 = Word2Vec(sentences=FGFS_abstracts, vector_size=100, window=2, min_count=0, workers=5, sg=0)
CBOW_embed_2.wv.save_word2vec_format('CBOW_word2vec_win_2.txt', binary=False)

CBOW_embed_4 = Word2Vec(sentences=FGFS_abstracts, vector_size=100, window=4, min_count=0, workers=5, sg=0)
CBOW_embed_4.wv.save_word2vec_format('CBOW_word2vec_win_4.txt', binary=False)

CBOW_embed_6 = Word2Vec(sentences=FGFS_abstracts, vector_size=100, window=6, min_count=0, workers=5, sg=0)
CBOW_embed_6.wv.save_word2vec_format('CBOW_word2vec_win_6.txt', binary=False)

CBOW_embed_8 = Word2Vec(sentences=FGFS_abstracts, vector_size=100, window=8, min_count=0, workers=5, sg=0)
CBOW_embed_8.wv.save_word2vec_format('CBOW_word2vec_win_8.txt', binary=False)

CBOW_embed_10 = Word2Vec(sentences=FGFS_abstracts, vector_size=100, window=10, min_count=0, workers=5, sg=0)
CBOW_embed_10.wv.save_word2vec_format('CBOW_word2vec_win_10.txt', binary=False)


##******************abstract CNN training***********************************************
print("create the tokenizer")
token = Tokenizer()  # create the tokenizer
token.fit_on_texts(FGFS_abstracts)  # fit the tokenizer on the documents
#print("Total words:", len(token.word_index))


word_index = token.word_index
#print('unique words: {}'.format(len(word_index)))


# # print()
vocab_size = len(token.word_index) + 1  # define vocabulary size (largest integer value)
#print('Vocabulary size: %d' % vocab_size)


#max_length = 259
max_length = max(len(l) for l in FGFS_abstracts) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
#print('샘플의 최대 길이 : {}'.format(max_length))


##=======================================train_valid data split====================================
train, test = train_test_split(FGFS_abstracts, test_size=0.30, random_state=1000)

print("train", len(train))
print("valid", len(test))


##====================================train_labels====================================
select_words = ['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
train_labels = []
for i in range(0, 3961):
    count = 0
    for j in range(0, len(select_words)):
        if select_words[j] in FGFS_abstracts[i]:
            count += 1
    if count >=1:
        train_labels.append(1)
    else:
        train_labels.append(0)


# ###====================================testation labels====================================
# select_words =['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
# valid_labels = []
# for i in range(0, 1359):
#
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in FGFS_abstracts[i]:
#             count += 1
#     if count >=1:
#         valid_labels.append(1)
#     else:
#         valid_labels.append(0)

###====================================testation labels====================================
select_words =['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
test_labels = []
for i in range(0, 1698):

    count = 0
    for j in range(0, len(select_words)):
        if select_words[j] in test[i]:
            count += 1
    if count >=1:
        test_labels.append(1)
    else:
        test_labels.append(0)


### ======================train, valid and test data encoding===================================
train_data = token.texts_to_sequences(train)
#valid_data = token.texts_to_sequences(valid)
test_data = token.texts_to_sequences(test)


X_train = pad_sequences(train_data, maxlen=max_length, padding='post')
y_train = np.asarray(train_labels).astype('float32').reshape((-1, 1))

#X_valid = pad_sequences(valid_data, max_length, padding='post')
#y_valid = np.asarray(valid_labels).astype('float32').reshape((-1, 1))

X_test = pad_sequences(test_data, max_length, padding='post')
y_test = np.asarray(test_labels).astype('float32').reshape((-1, 1))


##======================word2vec_matrix_win_2=====================================
embedding_index = {}
list_v = []
file = open('CBOW_word2vec_win_2.txt', 'rt', encoding='UTF8')
line = file.readline()
totalWords, numOfFeatures = line.split()
print(totalWords, numOfFeatures)
for line in file:
    values = line.split()
    list_v.append(values)
    word = values[0]
    coefs = array(values[1:], dtype='float64')
    embedding_index[word] = coefs


embedding_matrix_2 = np.array([[0 for col in range(100)] for row in range(12970)])

for word, i in token.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        if( i == 100):
            print(i,"번째 완료")
        for j in range(0, 100):
            embedding_matrix_2[i][j] = embedding_vector[j]


##======================word2vec_matrix_win_4=====================================
embedding_index = {}
list_v = []
file = open('CBOW_word2vec_win_4.txt', 'rt', encoding='UTF8')
line = file.readline()
totalWords, numOfFeatures = line.split()
print(totalWords, numOfFeatures)
for line in file:
    values = line.split()
    list_v.append(values)
    word = values[0]
    coefs = array(values[1:], dtype='float64')
    embedding_index[word] = coefs


embedding_matrix_4 = np.array([[0 for col in range(100)] for row in range(12970)])
for word, i in token.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        if( i == 100):
            print(i,"번째 완료")
        for j in range(0, 100):
            embedding_matrix_4[i][j] = embedding_vector[j]


## ======================word2vec_matrix_win_6=====================================
embedding_index = {}
list_v = []
file = open('CBOW_word2vec_win_6.txt', 'rt', encoding='UTF8')
line = file.readline()
totalWords, numOfFeatures = line.split()
print(totalWords, numOfFeatures)
for line in file:
    values = line.split()
    list_v.append(values)
    word = values[0]
    coefs = array(values[1:], dtype='float64')
    embedding_index[word] = coefs


embedding_matrix_6 = np.array([[0 for col in range(100)] for row in range(12970)])
for word, i in token.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        if( i == 100):
            print(i,"번째 완료")
        for j in range(0, 100):
            embedding_matrix_6[i][j] = embedding_vector[j]


## ======================word2vec_matrix_win_8=====================================
embedding_index = {}
list_v = []
file = open('CBOW_word2vec_win_8.txt', 'rt', encoding='UTF8')
line = file.readline()
totalWords, numOfFeatures = line.split()
print(totalWords, numOfFeatures)
for line in file:
    values = line.split()
    list_v.append(values)
    word = values[0]
    coefs = array(values[1:], dtype='float64')
    embedding_index[word] = coefs


embedding_matrix_8 = np.array([[0 for col in range(100)] for row in range(12970)])
for word, i in token.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        if( i == 100):
            print(i,"번째 완료")
        for j in range(0, 100):
            embedding_matrix_8[i][j] = embedding_vector[j]


## ======================word2vec_matrix_win_10=====================================
embedding_index = {}
list_v = []
file = open('CBOW_word2vec_win_10.txt', 'rt', encoding='UTF8')
line = file.readline()
totalWords, numOfFeatures = line.split()
print(totalWords, numOfFeatures)
for line in file:
    values = line.split()
    list_v.append(values)
    word = values[0]
    coefs = array(values[1:], dtype='float64')
    embedding_index[word] = coefs


embedding_matrix_10 = np.array([[0 for col in range(100)] for row in range(12970)])
for word, i in token.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        if( i == 100):
            print(i,"번째 완료")
        for j in range(0, 100):
            embedding_matrix_10[i][j] = embedding_vector[j]


##=======================CBOW_CNNs model using LSTM ====================================================
epochs = 100
embedding_dim = 100
pooling = 2
dropout = 0.2
filters_1 = 128
filters_2 = 256
filters_3 = 512
batch_sizes = 128
validation_splits = 0.33


##=======================CBOW_CNNs BiLSTM Win 2====================================================
cbow_cnn_win2_model = Sequential()
cbow_cnn_win2_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix_2], input_length=max_length, trainable=True))
cbow_cnn_win2_model.add(Conv1D(filters=filters_1, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_win2_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win2_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win2_model.add(Conv1D(filters=filters_2, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_win2_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win2_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win2_model.add(Conv1D(filters=filters_3, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_win2_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win2_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win2_model.add(Dense(10, activation='relu'))
cbow_cnn_win2_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_win2_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_win2_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')
CBOW_CNN_WIN_2 = cbow_cnn_win2_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])


CBOW_CNN_WIN_2_train = cbow_cnn_win2_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_WIN_2_train[1] * 100)))
CBOW_CNN_WIN_2_test = cbow_cnn_win2_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_WIN_2_test[1]*100)))


##====================F_score CBOW_cnn_Win_2==========================================
# predict probabilities for test set
CBOW_CNN_Win2_probs = cbow_cnn_win2_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_Win2_probs = CBOW_CNN_Win2_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_Win2_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_Win2_probs)))
print('CBOW_CNN_Win2_accuracy: %f' % CBOW_CNN_Win2_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_Win2_precision = precision_score(y_test, np.round(abs(CBOW_CNN_Win2_probs)))
print('CBOW_CNN_Win2_precision: %f' % CBOW_CNN_Win2_precision)
# recall: tp / (tp + fn)
CBOW_CNN_Win2_recall = recall_score(y_test, np.round(abs(CBOW_CNN_Win2_probs)))
print('CBOW_CNN_Win2_recall: %f' % CBOW_CNN_Win2_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_Win2_F1 = f1_score(y_test, np.round(abs(CBOW_CNN_Win2_probs)))
print('CBOW_CNN_Win2_F1: %f' % CBOW_CNN_Win2_F1)



##=======================CBOW_CNNs WIN_4 ====================================================
cbow_cnn_win4_model = Sequential()
cbow_cnn_win4_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix_4], input_length=max_length, trainable=True))
cbow_cnn_win4_model.add(Conv1D(filters=filters_1, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_win4_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win4_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win4_model.add(Conv1D(filters=filters_2, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_win4_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win4_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win4_model.add(Conv1D(filters=filters_3, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_win4_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win4_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win4_model.add(Dense(10, activation='relu'))
cbow_cnn_win4_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_win4_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_win4_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')

CBOW_CNN_WIN_4 = cbow_cnn_win4_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])


CBOW_CNN_WIN_4_train = cbow_cnn_win4_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_WIN_4_train[1] * 100)))
CBOW_CNN_WIN_4_test = cbow_cnn_win4_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_WIN_4_test[1]*100)))


##====================F_score CBOW_cnn_win 4==========================================
# predict probabilities for test set
CBOW_CNN_WIN4_probs = cbow_cnn_win4_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_WIN4_probs = CBOW_CNN_WIN4_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_WiN4_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_WIN4_probs)))
print('CBOW_CNN_WiN4_accuracy: %f' % CBOW_CNN_WiN4_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_WIN4_precision = precision_score(y_test, np.round(abs(CBOW_CNN_WIN4_probs)))
print('CBOW_CNN_WIN4_precision: %f' % CBOW_CNN_WIN4_precision)
# recall: tp / (tp + fn)
CBOW_CNN_WIN4_recall = recall_score(y_test, np.round(abs(CBOW_CNN_WIN4_probs)))
print('CBOW_CNN_WIN4_recall_recall: %f' % CBOW_CNN_WIN4_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_WIN4_F1 = f1_score(y_test, np.round(abs(CBOW_CNN_WIN4_probs)))
print('CBOW_CNN_WIN4_F1: %f' % CBOW_CNN_WIN4_F1)


##=======================CBOW_CNNs model using BiLSTM WIN 6====================================================
cbow_cnn_win6_model = Sequential()
cbow_cnn_win6_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix_6], input_length=max_length, trainable=True))
cbow_cnn_win6_model.add(Conv1D(filters=filters_1, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_win6_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win6_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win6_model.add(Conv1D(filters=filters_2, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_win6_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win6_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win6_model.add(Conv1D(filters=filters_3, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_win6_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win6_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win6_model.add(Dense(10, activation='relu'))
cbow_cnn_win6_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_win6_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_win6_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')

CBOW_CNN_WIN_6 = cbow_cnn_win6_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])


CBOW_CNN_WIN_6_train = cbow_cnn_win6_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_WIN_6_train[1] * 100)))
CBOW_CNN_WIN_6_Test = cbow_cnn_win6_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_WIN_6_Test[1]*100)))


##====================F_score CBOW_cnn_WIN 6==========================================
# predict probabilities for test set
CBOW_CNN_WIN6_probs = cbow_cnn_win6_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_WIN6_probs = CBOW_CNN_WIN6_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_WIN6_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_WIN6_probs)))
print('CBOW_CNN_WIN6_accuracy: %f' % CBOW_CNN_WIN6_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_WIN6_precision = precision_score(y_test, np.round(abs(CBOW_CNN_WIN6_probs)))
print('CBOW_CNN_WIN6_precision: %f' % CBOW_CNN_WIN6_precision)
# recall: tp / (tp + fn)
CBOW_CNN_WIN6_recall = recall_score(y_test, np.round(abs(CBOW_CNN_WIN6_probs)))
print('CBOW_CNN_WIN6_recall_recall: %f' % CBOW_CNN_WIN6_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_WIN6_F1 = f1_score(y_test, np.round(abs(CBOW_CNN_WIN6_probs)))
print('CBOW_CNN_WIN6_F1: %f' % CBOW_CNN_WIN6_F1)


##=======================CBOW_CNNs model using BiLSTM win 8 ====================================================
cbow_cnn_win8_model = Sequential()
cbow_cnn_win8_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix_8], input_length=max_length, trainable=True))
cbow_cnn_win8_model.add(Conv1D(filters=filters_1, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_win8_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win8_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win8_model.add(Conv1D(filters=filters_2, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_win8_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win8_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win8_model.add(Conv1D(filters=filters_3, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_win8_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win8_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win8_model.add(Dense(10, activation='relu'))
cbow_cnn_win8_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_win8_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_win8_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')
CBOW_CNN_WIN_8 = cbow_cnn_win8_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])


CBOW_CNN_WIN_8_train = cbow_cnn_win8_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_WIN_8_train[1] * 100)))
CBOW_CNN_WIN_8_Test = cbow_cnn_win8_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_WIN_8_Test[1]*100)))


##====================F_score CBOW_cnn_win 8==========================================
# predict probabilities for test set
CBOW_CNN_WIN8_probs = cbow_cnn_win8_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_WIN8_probs = CBOW_CNN_WIN8_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_WIN8_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_WIN8_probs)))
print('CBOW_CNN_WIN8_accuracy: %f' % CBOW_CNN_WIN8_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_WIN8_precision = precision_score(y_test, np.round(abs(CBOW_CNN_WIN8_probs)))
print('CBOW_CNN_WIN8_precision: %f' % CBOW_CNN_WIN8_precision)
# recall: tp / (tp + fn)
CBOW_CNN_WIN8_recall = recall_score(y_test, np.round(abs(CBOW_CNN_WIN8_probs)))
print('CBOW_CNN_WIN8_recall_recall: %f' % CBOW_CNN_WIN8_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_WIN8_F1 = f1_score(y_test, np.round(abs(CBOW_CNN_WIN8_probs)))
print('CBOW_CNN_WIN8_F1: %f' % CBOW_CNN_WIN8_F1)


##=======================CBOW_CNNs model using BiLSTM win 10====================================================
cbow_cnn_win10_model = Sequential()
cbow_cnn_win10_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix_10], input_length=max_length, trainable=True))
cbow_cnn_win10_model.add(Conv1D(filters=filters_1, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_win10_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win10_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win10_model.add(Conv1D(filters=filters_2, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_win10_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win10_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win10_model.add(Conv1D(filters=filters_3, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_win10_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_win10_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_win10_model.add(Dense(10, activation='relu'))
cbow_cnn_win10_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_win10_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_win10_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')

CBOW_CNN_WIN_10 = cbow_cnn_win10_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])


CBOW_CNN_WIN_10_train = cbow_cnn_win10_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_WIN_10_train[1] * 100)))
CBOW_CNN_WIN_10_Test = cbow_cnn_win10_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_WIN_10_Test[1]*100)))


##====================F_score CBOW_cnn_win 10==========================================
# predict probabilities for test set
CBOW_CNN_WIN10_probs = cbow_cnn_win10_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_WIN10_probs = CBOW_CNN_WIN10_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_WIN10_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_WIN10_probs)))
print('CBOW_CNN_WIN10_accuracy: %f' % CBOW_CNN_WIN10_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_WIN10_precision = precision_score(y_test, np.round(abs(CBOW_CNN_WIN10_probs)))
print('CBOW_CNN_WIN10_precision: %f' % CBOW_CNN_WIN10_precision)
# recall: tp / (tp + fn)
CBOW_CNN_WIN10_recall = recall_score(y_test, np.round(abs(CBOW_CNN_WIN10_probs)))
print('CBOW_CNN_WIN10_recall: %f' % CBOW_CNN_WIN10_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_WIN10_F1 = f1_score(y_test, np.round(abs(CBOW_CNN_WIN10_probs)))
print('CBOW_CNN_WIN10_F1: %f' % CBOW_CNN_WIN10_F1)

##=========================================CBOW_F1-Scores====================================
CBOW = [CBOW_CNN_Win2_F1, CBOW_CNN_WIN4_F1, CBOW_CNN_WIN6_F1, CBOW_CNN_WIN8_F1, CBOW_CNN_WIN10_F1]
print("All CBOW f1_score:", CBOW)


##================================CBOW model Acc figure========================================
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(CBOW_CNN_WIN_2.history['Accuracy'],  label="CBOW_WIN-2")
plt.plot(CBOW_CNN_WIN_4.history['Accuracy'],  label="CBOW_WIN-4")
plt.plot(CBOW_CNN_WIN_6.history['Accuracy'],  label="CBOW_WIN-6")
plt.plot(CBOW_CNN_WIN_8.history['Accuracy'],  label="CBOW_WIN-8")
plt.plot(CBOW_CNN_WIN_10.history['Accuracy'],  label="CBOW_WIN-10")
#plt.plot(CBOW_CNN_BiLSTM.history['val_Accuracy'], 'g', label="Validation")

# Label the plot.
plt.title("CBOW Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.00)
plt.legend(loc='lower right')
plt.savefig('CBOW_Word2vce_Win_Base_Acc(Kr_Confer)')
plt.show()

#
# #====================================F-Scores Result======================================
# title = ['Win_2', 'Win_4', 'Win_6', 'Win_8', 'Win_10']
# x = np.arange(len(title))
# # CBOW = [CBOW_lstm_f1, CBOW_gru_f1, CBOW_cnn_lstm_f1, CBOW_cnn_gru_f1]
# # Sg = [Sg_lstm_f1, Sg_gru_f1, Sg_cnn_lstm_f1, Sg_cnn_gru_f1]
#
# CBOW = [0.9012143091565473, 0.8756353778380209, 0.8989799276077657, 0.8869738150480609, 0.8648834019204389]
# Sg = [0.8825503355704699, 0.8938287989382879, 0.8667351832819459, 0.8975963121501481, 0.8865224625623961]
#
# # 넓이 지정
# width = 0.35
#
# # subplots 생성
# fig, axes = plt.subplots()
#
# # 넓이 설정
# axes.bar(x - width/2, CBOW, width, color='darkgrey', hatch='//', label='CNN-BiLSTM (CBOW)')
# axes.bar(x + width/2, Sg, width, color='black', label='CNN-BiLSTM (Sg)')
#
#
# # ticks & label 설정
# plt.xticks(x)
# axes.set_xticklabels(title)
# plt.xlabel('Models', fontweight='bold')
# plt.ylabel('F-Score', fontweight='bold')
# plt.legend(loc='upper left')
# plt.ylim(0, 1)
# #plt.grid()
# plt.savefig('KrConfer_F_score.png')