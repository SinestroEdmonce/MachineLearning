import numpy as np
from hmm import HMM
from collections import defaultdict


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)

    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    for idx, word in enumerate(unique_words.keys()):
        word2idx[word] = idx

    for idx, tag in enumerate(tags):
        tag2idx[tag] = idx

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))

    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    tag2word, tag2tag = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    initial, transition, tag_counter = defaultdict(int), defaultdict(int), defaultdict(int)
    for sentence in train_data:
        prev = None
        initial[sentence.tags[0]] += 1
        for i, word in enumerate(sentence.words):
            tag = sentence.tags[i]
            tag_counter[tag] += 1
            tag2word[tag][word] += 1
            tag2tag[prev][tag] += 1
            transition[prev] += 1
            prev = tag

    # Compute the parameters
    for tag in tags:
        pi[tag2idx[tag]] = initial[tag] / len(train_data)

        for next in tags:
            A[tag2idx[tag], tag2idx[next]] = (tag2tag[tag][next] / transition[tag]) if transition[tag] != 0 else 0

        for word, count in tag2word[tag].items():
            B[tag2idx[tag], word2idx[word]] = (count / tag_counter[tag]) if tag_counter[tag] != 0 else 0

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    S, index, expanded = len(tags), len(model.obs_dict), 0
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict:
                model.obs_dict[word] = index
                index += 1
                expanded += 1

    model.B = np.column_stack((model.B, np.full((S, expanded), 1e-6)))

    for sentence in test_data:
        tagging.append(model.viterbi(Osequence=sentence.words))

    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):
    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
