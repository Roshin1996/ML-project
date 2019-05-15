from jiwer import wer # used to calculate word error rate
import time
import sys

# data structures to implement HMM
observations = set() # set of English words
states = set() # set of French words
start_probability = dict() # Start probability of French words
transition_probability = dict() # Transition probability of French words
emission_probability = dict() # Emission probability of French to English words

# Performance metrics to evaluate HMM
training_time = -1
validation_time = -1
minWER = sys.maxsize # min word error rate when validating
maxWER = -1 # max word error rate when validating
totalWER = 0 # used to compute avg word error rate when validating
totalAccuracy = 0 # used to compute avg accuracy when validating


# Viterbi decoding algorithm to translate English sentence into French sentence
# Input: obs - English sentence
# Output: max_path - French sentence
def viterbi_decode(obs):
    path = {s: [] for s in states}  # init path: path[s] represents the path ends with s
    current_probability = {}
    for s in states:
        current_probability[s] = start_probability[s] * emission_probability[s][obs[0]]
    for i in range(1, len(obs)):
        last_probability = current_probability
        current_probability = {}
        for current_state in states:
            max_probability, ls = max(
                ((last_probability[last_state] * transition_probability[last_state][current_state]
                  * emission_probability[current_state][obs[i]], last_state)
                 for last_state in states))
            current_probability[current_state] = max_probability
            path[current_state].append(ls)

    # find final largest probability
    max_probability = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if current_probability[s] > max_probability:
            max_path = path[s]
            max_probability = current_probability[s]
    return max_path


# Train HMM using data from English and French input text files
def train():

    print('Training start.')

    start_time = time.time()

    # open French text file
    fo = open('small_vocab_fr.txt', encoding='utf8')

    # open English text file and iterate line by line
    for enLine in open('small_vocab_en.txt', encoding='utf8'):

        # read French line
        frLine = fo.readline()

        # split English and French sentence into array of words
        enTokens = enLine.strip().split(' ')
        enTokensLength = len(enTokens)
        frTokens = frLine.strip().split(' ')
        frTokensLength = len(frTokens)

        # Update start prob matrix
        start_probability[frTokens[0]] = start_probability.get(frTokens[0], 0) + 1

        # Add English words in current line to set of observations
        for i in range(0, enTokensLength):
            if enTokens[i] != '':
                observations.add(enTokens[i])

        # Add French words in current line to set of states. Update transition and emission matrices.
        for i in range(0, frTokensLength):
            if frTokens[i] != '':
                states.add(frTokens[i])

                if frTokens[i] not in transition_probability:
                    transition_probability[frTokens[i]] = dict()

                if i == frTokensLength - 1:
                    nextToken = 'END'
                else:
                    nextToken = frTokens[i + 1]

                transition_probability[frTokens[i]][nextToken] = transition_probability[frTokens[i]]\
                                                                     .get(nextToken, 0) + 1

                if frTokens[i] not in emission_probability:
                    emission_probability[frTokens[i]] = dict()

                if i <= enTokensLength - 1:
                    emission_probability[frTokens[i]][enTokens[i]] = emission_probability[frTokens[i]]\
                                                                         .get(enTokens[i], 0) + 1

    # Fill in remaining values for start prob, transition, and emission matrices with zeros
    for state in states:
        if state not in start_probability.keys():
            start_probability[state] = 0
        for key, value in transition_probability.items():
            if state not in value.keys():
                transition_probability[key][state] = 0

    for ob in observations:
        for key, value in emission_probability.items():
            if ob not in value.keys():
                emission_probability[key][ob] = 0

    # normalize start probability, transition probability, and emission probability matrices
    # into probability distributions
    start_probability_total = sum(start_probability.values())
    for token, count in start_probability.items():
        start_probability[token] = count / start_probability_total

    for prev_word, next_word_list in transition_probability.items():
        row_total = sum(next_word_list.values())
        if row_total != 0:
            for token, count in next_word_list.items():
                next_word_list[token] = count / row_total
            transition_probability[prev_word] = next_word_list

    for word, emitted_word_list in emission_probability.items():
        row_total = sum(emitted_word_list.values())
        if row_total != 0:
            for token, count in emitted_word_list.items():
                emitted_word_list[token] = count / row_total
            emission_probability[word] = emitted_word_list

    # compute training time
    global training_time
    training_time = time.time() - start_time

    print('Training successful. Training time: ', training_time * 1000, ' ms')


# Validate HMM using data from first 100 lines of English and French files
def validate(validation_size):

    print('Validation start.')

    start_time = time.time()

    lineNum = 1
    validation_acc_sum = 0

    # Open French file. Start from beginning.
    fo = open('small_vocab_fr.txt', encoding='utf8')

    # Open English file at beginning. Iterate line by line.
    for enLine in open('small_vocab_en.txt', encoding='utf8'):

        # Stop after validating the specified number of lines
        if lineNum == validation_size + 1:
            break

        print('---------------------------------')
        print('Validating Line ', lineNum)

        # Read French line
        frLine = fo.readline()

        # Split English and French sentence into tokens
        frTokens = frLine.strip().split(' ')
        enTokens = enLine.strip().split(' ')

        # Translate English to French sentence using HMM and Viterbi decoding
        viterbi_result = viterbi_decode(enTokens)

        # compute word error rate. reference = frTokens, hypothesis = viterbi_result
        wer_value = wer(frTokens, viterbi_result)

        global minWER, maxWER, totalWER, totalAccuracy

        # find min, max, and total WER so far
        if wer_value < minWER:
            minWER = wer_value

        if wer_value > maxWER:
            maxWER = wer_value

        totalWER += wer_value

        # compute translation accuracy of current line based on WER. accuracy >= 0.
        accuracy = max(1 - wer_value, 0)

        totalAccuracy += accuracy

        print('Desired Result: ', ' '.join(frTokens))
        print('Actual Result : ', ' '.join(viterbi_result))
        print('Word Error Rate: ', wer_value)
        print('Translation Accuracy: ', accuracy * 100, ' %')

        lineNum += 1

    # compute validation time
    global validation_time
    validation_time = time.time() - start_time

    print('Validation complete.')

if __name__ == '__main__':

    # Train HMM with entire input data
    train()

    # Validate HMM with data from first 100 lines of each file
    validation_size = 100
    validate(validation_size)

    # Print summary
    print('---------------------------------')
    print('Summary')
    print()
    print('Training Time: ', training_time * 1000, ' ms')
    print('Validation Time: ', validation_time * 1000, ' ms')
    print('Min WER: ', minWER)
    print('Max WER: ', maxWER)
    print('Average WER: ', totalWER / validation_size)
    print('Average Translation Accuracy: ', totalAccuracy / validation_size * 100, ' %')