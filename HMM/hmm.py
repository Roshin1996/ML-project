from jiwer import wer
import time


# data structures to implement HMM
observations = set() # set of English words
states = set() # set of French words
start_probability = dict() # Start probability of French words
transition_probability = dict() # Transition probability of French words
emission_probability = dict() # Emission probability of French to English words


'''
# convert list of counts into list of probability
def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict
'''


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

    # find the final largest probability
    max_probability = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if current_probability[s] > max_probability:
            max_path = path[s]
            max_probability = current_probability[s]
    return max_path


training_time = -1
validation_time = -1

# Train HMM using data from English and French input text files
def train():

    start_time = time.time()

    fo = open('small_vocab_fr.txt', encoding='utf8')

    for enLine in open('small_vocab_en.txt', encoding='utf8'):

        frLine = fo.readline()

        enTokens = enLine.strip().split(' ')
        enTokensLength = len(enTokens)
        frTokens = frLine.strip().split(' ')
        frTokensLength = len(frTokens)

        start_probability[frTokens[0]] = start_probability.get(frTokens[0], 0) + 1

        for i in range(0, enTokensLength):
            if enTokens[i] != '':
                observations.add(enTokens[i])

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

    '''
    goodS = 0
    goodT = 0
    goodS2 = 0
    goodT2 = 0
    goodE = 0
    goodE2 = 0

    for k, v in start_probability.items():
        if v > 0:
            # print('goodS')
            # print(v)
            goodS += 1

    for k, v in transition_probability.items():
        for a, b in v.items():
            if b > 0:
                # print('goodT')
                # print(b)
                goodT += 1

    for k, v in emission_probability.items():
        for a, b in v.items():
            if b > 0:
                # print('goodT')
                # print(b)
                goodE += 1

    # print(start_probability)
    # print(transition_probability)

    # exit(0)
    '''

    # normalize start probability, transition probability, and emission probability matrices
    # into probability distributions
    start_probability_total = sum(start_probability.values())
    for key, value in start_probability.items():
        start_probability[key] = value / start_probability_total

    for prev_word, next_word_list in transition_probability.items():
        s = sum(next_word_list.values())
        if s == 0:
            continue
        for k, v in next_word_list.items():
            next_word_list[k] = v / s
        # transition_probability[prev_word] = list2probabilitydict(next_word_list)
        transition_probability[prev_word] = next_word_list

    for prev_word, next_word_list in emission_probability.items():
        s = sum(next_word_list.values())
        if s == 0:
            continue
        for k, v in next_word_list.items():
            next_word_list[k] = v / s
        # emission_probability[prev_word] = list2probabilitydict(next_word_list)
        emission_probability[prev_word] = next_word_list

    '''
    for k, v in start_probability.items():
        if v > 0:
            # print('goodS')
            # print(v)
            goodS2 += 1

    for k, v in transition_probability.items():
        for a, b in v.items():
            if b > 0:
                # print('goodT')
                # print(b)
                goodT2 += 1

    for k, v in emission_probability.items():
        for a, b in v.items():
            if b > 0:
                # print('goodT')
                # print(b)
                goodE2 += 1

    print(goodS, ' ', goodS2, ' ', goodT, ' ', goodT2, ' ', goodE, ' ', goodE2)

    assert (goodS == goodS2)
    assert (goodT == goodT2)
    assert (goodE == goodE2)

    # print(observations)
    # print(states)
    # print(start_probability)
    # print(transition_probability)
    # print(emission_probability)

    # exit(0)

    # print(len(observations))
    # print(len(states))
    # print(len(start_probability))
    # print(len(transition_probability))
    # for k, v in transition_probability.items():
    #    print('T ', len(v))
    # print(len(emission_probability))
    # for k, v in emission_probability.items():
    #    print('E ', len(v))
    '''

    global training_time
    training_time = time.time() - start_time


minWER = 1000
maxWER = -1
totalWER = 0

def validate():
    start_time = time.time()

    ct = 0
    validation_acc_sum = 0

    fo = open('small_vocab_fr.txt', encoding='utf8')
    for enLine in open('small_vocab_en.txt', encoding='utf8'):

        print(ct)

        #if ct < training_size:
        #    ct += 1
        #    continue

        if ct == 100:
            break

        #print('Validation ', ct)

        frLine = fo.readline()
        frTokens = frLine.strip().split(' ')
        print(frTokens)
        enTokens = enLine.strip().split(' ')
        #ans = viterbi_decode(enTokens, states, start_probability, transition_probability, emission_probability)
        ans = viterbi_decode(enTokens)
        #print(ans)
        #if ct == training_size:
        #    print(frTokens)
        #    print(ans)
        werx = wer(frTokens, ans)

        global minWER, maxWER, totalWER, validation_time

        if werx < minWER:
            minWER = werx

        if werx > maxWER:
            maxWER = werx

        totalWER = totalWER + werx

        print(werx)
        if werx < 1.0:
            print(frTokens)
            print(ans)
            #print(werx - len(frTokens))
            #print(werx - len(ans))
        print('----------')
            #wacc = 1 - werx
            #if wacc < 0:
            #    wacc = 0
            #print(wer)
            #validation_acc_sum += wacc

        ct += 1

    #validation_accuracy = validation_acc_sum / ct

    validation_time = time.time() - start_time

    #print(validation_accuracy)


if __name__ == '__main__':

    # Train HMM with entire input data
    train()

    # Validate HMM with data from first 100 lines of each file
    validate()

    # Print summary
    print('Training Time: ', training_time * 1000, ' ms')
    print('Validation Time: ', validation_time * 1000, ' ms')
    print('Min WER: ', minWER)
    print('Max WER: ', maxWER)
    print('Average WER: ', totalWER / 100)