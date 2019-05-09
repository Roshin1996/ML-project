import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils,to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from pickle import dump
from sys import argv
from os import makedirs
import shutil

tokenizer=Tokenizer()
sequences=open('data/formatted_text.txt').readlines()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
sequences=np.array(sequences)
vocab = len(tokenizer.word_index) + 1
dump(tokenizer, open('tokenizer/tokenizer.pkl', 'wb'))


input_X=[]
output_Y=[]
input_X, output_Y = sequences[:,:-1], sequences[:,-1]
output_Y=to_categorical(output_Y, num_classes=vocab)
	
ipdim= input_X.shape[1]

model = Sequential()
model.add(Embedding(vocab, 300, input_length=ipdim))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

shutil.rmtree('models')
makedirs('models')
filepath='models/my_model.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
callbacks_list = [checkpoint]

model.fit(input_X, output_Y, epochs=200, batch_size=64, callbacks=callbacks_list)



