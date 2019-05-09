import string
import numpy as np
import sys
import time
from random import randint

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','', string.punctuation))

def add2dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict

# Probability dictionaries
initial_word = {} # P(initial word)
second_word = {} # P(second word | initial word)
transitions = {} # P(next word | (initial word, second word))

# Vars
training_data_file = sys.argv[1]
number_of_words = int(sys.argv[2])

training_lines = 0
training_words = 0
training_time = 0

generated_words = 0
generation_time = 0
n_text=0
line=""
tokens=[]


# Trains a Markov model based on the data in training_data_file
def train_markov_model():

	global training_data_file, training_lines, training_words, training_time,n_text,line,tokens
	print('Training with file ', training_data_file)

	start_time = time.time()
	line=open(training_data_file, encoding="utf8").read()

	#for line in open(training_data_file, encoding="utf8"):
	tokens = remove_punctuation(line.rstrip().lower()).split()
	tokens_length = len(tokens)
	n_text=len(tokens)
	training_lines += 1
	training_words += tokens_length
	for i in range(tokens_length):
		token = tokens[i]
		if i == 0:
			initial_word[token] = initial_word.get(token, 0) + 1
		else:
			prev_token = tokens[i - 1]
			if i == tokens_length - 1:
				add2dict(transitions, (prev_token, token), 'END')
			if i == 1:
				add2dict(second_word, prev_token, token)
			else:
				prev_prev_token = tokens[i - 2]
				add2dict(transitions, (prev_prev_token, prev_token), token)

    # Normalize the distributions
	initial_word_total = sum(initial_word.values())
	for key, value in initial_word.items():
		initial_word[key] = value / initial_word_total

	for prev_word, next_word_list in second_word.items():
		second_word[prev_word] = list2probabilitydict(next_word_list)
        #print(second_word[prev_word])

	for word_pair, next_word_list in transitions.items():
		transitions[word_pair] = list2probabilitydict(next_word_list)
        #print(transitions[word_pair])

	training_time = time.time() - start_time

	print('Training successful.')


# Function to randomly sample word from dictionary (initial_word, second_word, transitions)
def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key
    assert(False)


# Function to generate sample text
def generate():
	global number_of_words, generated_words, generation_time

	print('Generating sample text. Number of Words = ', number_of_words)
	print('---------------------------------')

	start_time = time.time()

	#for i in range(number_of_words):
	sentence = []

	# Initial word
	r=randint(0,n_text-2)
	word0 = tokens[r]
	sentence.append(word0)

	# Second word
	word1 = tokens[r+1]
	sentence.append(word1)

	# Subsequent words until END
	for i in range(number_of_words):
		word2 = sample_word(transitions[(word0, word1)])
		sentence.append(word2)
		word0 = word1
		word1 = word2

	# Join array of words into sentence string
	print(' '.join(sentence))
	#generated_words += len(sentence)

	generation_time = time.time() - start_time

	print('---------------------------------')
	print('Text generated.')


if __name__=='__main__':
    train_markov_model()
    generate()
    print('\nSummary:')
    print('Training Data Size: ', training_words, ' words')
    print('Training Time: ', round(training_time * 1000, 6), ' ms')
    #print('Text Generation Data Size: ', number_of_words, ' words')
    print('Text Generation Time: ', round(generation_time * 1000, 6), ' ms')
    #print('Accuracy: ', accuracy * 100, '%')