import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
 
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# intefr encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
 
# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units,embmat):
	model = Sequential()
	model.add(Embedding(src_vocab, 100, weights=[embmat],input_length=src_timesteps, mask_zero=True,trainable=False))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model
 
# load datasets
dataset = pickle.load(open('data/full.pkl','rb'))
train = pickle.load(open('data/train.pkl','rb'))
test = pickle.load(open('data/test.pkl','rb'))
 
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max(len(line.split()) for line in dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))


fr_tokenizer = create_tokenizer(dataset[:, 1])
fr_vocab_size = len(fr_tokenizer.word_index) + 1
fr_length = max(len(line.split()) for line in dataset[:, 1])
print('french Vocabulary Size: %d' % fr_vocab_size)
print('french Max Length: %d' % (fr_length))

with open('tokenizer/english_tokenizer.pkl','wb') as f:
	pickle.dump([eng_tokenizer,eng_vocab_size,eng_length],f)
with open('tokenizer/french_tokenizer.pkl','wb') as f:
	pickle.dump([fr_tokenizer,fr_vocab_size,eng_length],f)

embeddings_index = {}
f = open('glove.6B/glove.6B.100d.txt',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((eng_vocab_size, 100))
for word, i in eng_tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
 
# prepare training data
trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 1])
trainY = encode_sequences(fr_tokenizer, fr_length, train[:, 0])
trainY = encode_output(trainY, fr_vocab_size)
# prepare validation data
testX = encode_sequences(eng_tokenizer, eng_length, test[:, 1])
testY = encode_sequences(fr_tokenizer, fr_length, test[:, 0])
testY = encode_output(testY, fr_vocab_size)

with open('tokenizer/encoded_data.pkl','wb') as f:
	pickle.dump([trainX,testX],f)


 
# define model
model = define_model(eng_vocab_size, fr_vocab_size, eng_length, fr_length, 256,embedding_matrix)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# summarize defined model
print(model.summary())
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint])