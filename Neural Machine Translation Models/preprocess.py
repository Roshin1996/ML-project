import string
import re
import pickle
from unicodedata import normalize
import numpy as np

 
def clean_pairs(lines):
	cleaned = list()
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			line = line.split()
			line = [word.lower() for word in line]
			line = [re_print.sub('', w) for w in line]
			line = [word for word in line if word.isalpha()]
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return np.array(cleaned)
 
engfile = 'data/english.txt'
frfile='data/french.txt'

english_text=open(engfile, mode='rt', encoding='utf-8').read()
french_text=open(frfile, mode='rt', encoding='utf-8').read()
	
english_lines = english_text.strip().split('\n')
french_lines=french_text.strip().split('\n')

l=len(english_lines)

pairs=[[english_lines[i],french_lines[i]] for i in range(l)]

clean_pairs = clean_pairs(pairs)
	
size = 10000
dataset = clean_pairs[:size, :]
np.random.shuffle(dataset)
l=int(0.9*size)
train, test = dataset[:l], dataset[l:]

with open('data/full.pkl','wb') as f:
	pickle.dump(dataset,f)
with open('data/train.pkl','wb') as f:
	pickle.dump(train,f)
with open('data/test.pkl','wb') as f:
	pickle.dump(test,f)
print('Saved training and testing data.')
