"""
intro2nlp, assignment 3, 2021

In this assignment you will implement a Hidden Markov model
to predict the part of speech sequence for a given sentence.

"""


from math import log, isfinite
from collections import Counter

import sys, os, time, platform, nltk, random

def main():
    train_sentences = load_annotated_corpus('en-ud-train.upos.tsv')
    learn_params(train_sentences)
    test_sentences = load_annotated_corpus('en-ud-dev.upos.tsv')
    predicted_tags = []
    for sentence in test_sentences:
        predicted_tags.append(baseline_tag_sentence([word for word, tag in sentence], perWordTagCounts, allTagCounts))

    right_predictions = 0
    for test_sentence, predicted_sentence in zip(test_sentences, predicted_tags):
        for (test_word, test_tag), (predicted_word, predicted_tag) in zip(test_sentence, predicted_sentence):
            if test_tag == predicted_tag:
                right_predictions += 1
    print(f'Accuracy is {right_predictions / sum([len(sentence) for sentence in test_sentences])}')
    stop = 0


# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    #TODO edit the dictionary to have your own details
    return {'name': 'Maxim Zhivodrov', 'id': '317649606', 'email': 'maximzh@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
emissionCounts = {}
transitionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities

def learn_params(tagged_sentences):
    """
    Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
       and emissionCounts data-structures.
      allTagCounts and perWordTagCounts should be used for baseline tagging and
      should not include pseudocounts, dummy tags and unknowns.
      The transisionCounts and emmisionCounts
      should be computed with pseudo tags and shoud be smoothed.
      A and B should be the log-probability of the normalized counts, based on
      transisionCounts and  emmisionCounts

    Args:
        tagged_sentences: a list of tagged sentences, each tagged sentence is a
        list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
        [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    #TODO complete the code

    global START,END, UNK, allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B

    all_tags = []
    all_words = []
    for sentence in tagged_sentences:
        for word, tag in sentence:
            all_words.append(word)
            all_tags.append(tag)

    set_all_words = set(all_words)
    set_all_tags = set(all_tags)
    dict_all_tags = dict(Counter([tag for sentence in tagged_sentences for word,tag in sentence]))
    dict_end_tag = {tag:0 for tag in set_all_tags}
    for sentence in tagged_sentences:
        dict_end_tag[sentence[-1][1]] += 1

    #allTagsCount population
    allTagCounts.update(all_tags)

    #perWordTagCounts population
    for word in all_words:
        perWordTagCounts[word] = {}
    for sentence in tagged_sentences:
        for word, tag in sentence:
            if tag in perWordTagCounts[word]:
                perWordTagCounts[word][tag] += 1
            else:
                perWordTagCounts[word][tag] = 0

    #transitionCounts population
    for tag1 in set_all_tags:
        transitionCounts[f'{START}+{tag1}'] = 0
        transitionCounts[f'{tag1}+{END}'] = 0
        for tag2 in set_all_tags:
            transitionCounts[f'{tag1}+{tag2}'] = 0

    for sentence in tagged_sentences:
        transitionCounts[f'{START}+{sentence[0][1]}'] += 1
        transitionCounts[f'{sentence[-1][1]}+{END}'] += 1
        for index in range(len(sentence)-1):
            tag_tag_combo = f'{sentence[index][1]}+{sentence[index+1][1]}'
            transitionCounts[tag_tag_combo] += 1

    #emissionCounts population
    for tag in set_all_tags:
        for word in set_all_words:
            emissionCounts[f'{tag}+{word}'] = 0
    for word in set_all_words:
        emissionCounts[f'{START}+{word}'] = 0
    for sentence in tagged_sentences:
        first_word = sentence[0][0]
        emissionCounts[f'{START}+{first_word}'] += 1
        for word,tag in sentence:
            tag_word_combo = f'{tag}+{word}'
            emissionCounts[tag_word_combo] += 1

    #A population
    for tag_tag_combo,count in transitionCounts.items():
        first_tag,second_tag = tag_tag_combo.split('+')[0], tag_tag_combo.split('+')[1]
        if first_tag == START:
            if count != 0:
                A[tag_tag_combo] = count / len(tagged_sentences)
            else:
                A[tag_tag_combo] = (count + 1) / (len(tagged_sentences) + len(set_all_tags))
        elif second_tag == END:
            if count != 0:
                try: A[tag_tag_combo] = count / dict_end_tag[first_tag]
                except ZeroDivisionError: A[tag_tag_combo] = (count+1) / len(set_all_tags)
            else:
                A[tag_tag_combo] = (count+1) / (dict_end_tag[first_tag] + len(set_all_tags))
        else:
            if count != 0:
                A[tag_tag_combo] = count / dict_all_tags[first_tag]
            else:
                A[tag_tag_combo] = (count + 1) / (dict_all_tags[first_tag] + len(set_all_tags))

    #B population
    for tag_word_combo, count in emissionCounts.items():
        tag, word = tag_word_combo.split('+')[0], tag_word_combo.split('+')[1]
        if tag == START:
            if count != 0:
                B[tag_word_combo] = count / len(tagged_sentences)
            else:
                B[tag_word_combo] = (count+1) / (len(tagged_sentences) + len(set_all_tags))
        else:
            if count != 0:
                B[tag_word_combo] = count / dict_all_tags[tag]
            else:
                B[tag_word_combo] = (count + 1) / (dict_all_tags[tag] + len(set_all_tags))



    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]

def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """

    #TODO complete the code
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts:
            tagged_sentence.append((word, max(perWordTagCounts[word], key=perWordTagCounts[word].get)))
        else:
            # sample = random.choice(list(allTagCounts.keys()), weights = list(allTagCounts.values()))
            sample = random.choices(list(allTagCounts.keys()), weights = list(allTagCounts.values()), k = 1)[0]
            tagged_sentence.append((word,sample))

    return tagged_sentence

#===========================================
#       POS tagging with HMM
#===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    #TODO complete the code
    tagged_sentence = []
    return tagged_sentence

def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
        # Hint 1: For efficiency reasons - for words seen in training there is no
        #      need to consider all tags in the tagset, but only tags seen with that
        #      word. For OOV you have to consider all tags.
        # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
        #         current list = [ the dummy item ]
        # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END


    #TODO complete the code
    v_last = None
    return v_last

#a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    pass

#a suggestion for a helper function. Not an API requirement
def predict_next_best(word, tag, predecessor_list):
    """
    Returns a new item (tupple)
    """
    pass



def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags

    #TODO complete the code

    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}


        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.


    Return:
        list: list of pairs
    """
    if list(model.keys())[0]=='baseline':
        return baseline_tag_sentence(sentence, model.values()[0], model.values()[1])
    if list(model.keys())[0]=='hmm':
        return hmm_tag_sentence(sentence, model.values()[0], model.values()[1])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence)==len(pred_sentence)

    #TODO complete the code
    correct, correctOOV, OOV = None, None, None
    return correct, correctOOV, OOV


if __name__ == '__main__':
    main()