from jiwer import wer
import time

# data structures to implement HMM
observations = set()
states = set()
start_probability = dict()
transition_probability = dict()
emission_probability = dict()


# convert list of counts into list of probability
def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict


def Viterbit(obs, states, s_pro, t_pro, e_pro):
    path = {s: [] for s in states}  # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in states:
        #print(curr_pro[s])
        #print(s_pro[s])
        #print(e_pro[s])
        #print(obs[0])
        #print(e_pro[s][obs[0]])
        curr_pro[s] = s_pro[s] * e_pro[s][obs[0]]
    for i in range(1, len(obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:
            max_pro, last_sta = max(
                ((last_pro[last_state] * t_pro[last_state][curr_state] * e_pro[curr_state][obs[i]], last_state)
                 for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    # find the final largest probability
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
            # print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path




if __name__ == '__main__':
    validation_split = 0.2
    training_size = 137860 * 0.8
    validation_size = 1 - training_size
    #obs = ['new', 'jersey', 'is', 'sometimes', 'quiet', 'during', 'autumn']
    #obs = ['normal', 'cold', 'dizzy']
    #tokens, tokens2 = []
    #states.add('END')

    start_time = time.time()

    fo = open('small_vocab_fr.txt', encoding='utf8')
    ln = 0
    for enLine in open('small_vocab_en.txt', encoding='utf8'):
        ln += 1


        frLine = fo.readline()

        enTokens = enLine.strip().split(' ')
        #print(enTokens)
        enTokensLength = len(enTokens)
        frTokens = frLine.strip().split(' ')
        frTokensLength = len(frTokens)

        start_probability[frTokens[0]] = start_probability.get(frTokens[0], 0) + 1

        for i in range(0, enTokensLength):
            #enTokens[i] = enTokens[i].strip()
            if enTokens[i] == '':
                continue

            observations.add(enTokens[i])

        for i in range(0, frTokensLength):
            #frTokens[i] = frTokens[i].strip()
            if frTokens[i] == '':
                continue

            states.add(frTokens[i])

            if frTokens[i] not in transition_probability:
                transition_probability[frTokens[i]] = dict()

            if i == frTokensLength - 1:
                nextToken = 'END'
            else:
                nextToken = frTokens[i + 1]
                #nextToken = nextToken.strip()

            transition_probability[frTokens[i]][nextToken] = transition_probability[frTokens[i]].get(nextToken, 0) + 1

            if frTokens[i] not in emission_probability:
                emission_probability[frTokens[i]] = dict()

            if i <= enTokensLength - 1:
                emission_probability[frTokens[i]][enTokens[i]] = emission_probability[frTokens[i]].get(enTokens[i], 0) + 1

    training_time = time.time() - start_time

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

    goodS = 0
    goodT = 0
    goodS2 = 0
    goodT2 = 0
    goodE = 0
    goodE2 = 0

    for k, v in start_probability.items():
        if v > 0:
            #print('goodS')
            #print(v)
            goodS += 1

    for k, v in transition_probability.items():
        for a, b in v.items():
            if b > 0:
                #print('goodT')
                #print(b)
                goodT += 1

    for k, v in emission_probability.items():
        for a, b in v.items():
            if b > 0:
                # print('goodT')
                #print(b)
                goodE += 1

    #print(start_probability)
    #print(transition_probability)

    #exit(0)

    #normalize into prob dist
    start_probability_total = sum(start_probability.values())
    for key, value in start_probability.items():
        start_probability[key] = value / start_probability_total

    for prev_word, next_word_list in transition_probability.items():
        s = sum(next_word_list.values())
        if s == 0:
            continue
        for k, v in next_word_list.items():
            next_word_list[k] = v / s
        #transition_probability[prev_word] = list2probabilitydict(next_word_list)
        transition_probability[prev_word] = next_word_list

    for prev_word, next_word_list in emission_probability.items():
        s = sum(next_word_list.values())
        if s == 0:
            continue
        for k, v in next_word_list.items():
            next_word_list[k] = v / s
        #emission_probability[prev_word] = list2probabilitydict(next_word_list)
        emission_probability[prev_word] = next_word_list

    for k, v in start_probability.items():
        if v > 0:
            #print('goodS')
            #print(v)
            goodS2 += 1

    for k, v in transition_probability.items():
        for a, b in v.items():
            if b > 0:
                #print('goodT')
                #print(b)
                goodT2 += 1

    for k, v in emission_probability.items():
        for a, b in v.items():
            if b > 0:
                #print('goodT')
                #print(b)
                goodE2 += 1

    print(goodS, ' ', goodS2, ' ', goodT, ' ', goodT2, ' ', goodE, ' ', goodE2)

    assert(goodS == goodS2)
    assert(goodT == goodT2)
    assert(goodE == goodE2)

    #print(observations)
    #print(states)
    #print(start_probability)
    #print(transition_probability)
    #print(emission_probability)

    #exit(0)

    #print(len(observations))
    #print(len(states))
    #print(len(start_probability))
    #print(len(transition_probability))
    #for k, v in transition_probability.items():
    #    print('T ', len(v))
    #print(len(emission_probability))
    #for k, v in emission_probability.items():
    #    print('E ', len(v))

    start_time = time.time()

    ct = 0
    validation_acc_sum = 0
    minWER = 1000
    maxWER = -1
    totalWER = 0
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
        ans = Viterbit(enTokens, states, start_probability, transition_probability, emission_probability)
        #print(ans)
        #if ct == training_size:
        #    print(frTokens)
        #    print(ans)
        werx = wer(frTokens, ans)

        if werx < minWER:
            minWER = werx

        if werx > maxWER:
            maxWER = werx

        totalWER += werx

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

    print('Training Time: ', training_time * 1000, ' ms')
    print('Validation Time: ', validation_time * 1000, ' ms')
    print('Min WER: ', minWER)
    print('Max WER: ', maxWER)
    print('Average WER: ', totalWER / 100)