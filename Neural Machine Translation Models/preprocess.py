import string
import re
from pickle import dump
from unicodedata import normalize
import numpy as np

 
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return np.array(cleaned)
 
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)
 
filename = 'data/fra.txt'
with open(filename, mode='rt', encoding='utf-8') as f:
	text=f.read()
	
lines = text.strip().split('\n')
pairs = [line.split('\t') for line in  lines]

clean_pairs = clean_pairs(pairs)
	
size = 10000
dataset = clean_pairs[:size, :]
np.random.shuffle(dataset)
l=int(0.9*size)
train, test = dataset[:l], dataset[l:]

with open('english-french-both.pkl','wb') as f:
	dump(dataset,f)
with open('english-french-train.pkl','wb') as f:
	dump(train,f)
with open('english-french-test.pkl','wb') as f:
	dump(test,f)
print('Saved training and testing data.')
