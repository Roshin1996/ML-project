import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# map an intefr to a word
def word_for_id(intefr, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == intefr:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	intefrs = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in intefrs:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 
# load datasets
dataset = pickle.load(open('data/full.pkl','rb'))
train = pickle.load(open('data/train.pkl','rb'))
test = pickle.load(open('data/test.pkl','rb'))

eng_tokenizer,eng_vocab_size,eng_length=pickle.load(open('tokenizer/english_tokenizer.pkl','rb'))
fr_tokenizer,fr_vocab_size,fr_length=pickle.load(open('tokenizer/french_tokenizer.pkl','rb'))

trainX, testX = pickle.load(open('tokenizer/encoded_data.pkl','rb'))

model = load_model('model.h5')
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)
print('test')
evaluate_model(model, eng_tokenizer, testX, test)