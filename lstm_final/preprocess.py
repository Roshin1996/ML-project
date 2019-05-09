import numpy as np
from pickle import dump


filename = "data/input_text.txt"
text = open(filename,encoding='utf-8').read()
for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n':
    text=text.replace(ch," ")
tokenized_text=text.split()
tokenized_text = [word for word in tokenized_text if word.isalpha()]


n_words=len(tokenized_text)
seq_length=50
sequences = []
for i in range(n_words-seq_length-1):
	seq = tokenized_text[i:i+seq_length+1]
	line = ' '.join(seq)
	sequences.append(line)

def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()	


save_doc(sequences,'data/formatted_text.txt')