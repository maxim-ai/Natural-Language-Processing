import math
import random
import datetime
from collections import Counter


class Spell_Checker:

    def __init__(self, lm = None):
        self.lm = lm
        self.et = None

    def add_language_model(self, lm):
        self.lm = lm

    def build_model(self, text, n):
        pass

    def add_error_tables(self, error_tables):
        self.et = error_tables

    def spell_check(self, text, alpha):
        # Should input text be normalized in the method or given normalized? !!!
        str_parts = text.split()
        if self.check_if_has_wrong_word(str_parts) is not None:
            pass
        else:
            pass

    def evaluate(self, text):
        return self.lm.evaluate(text)

    # My methods
    def check_if_has_wrong_word(self,str_parts):
        WORDS = self.lm.WORDS
        for word in str_parts:
            if word not in WORDS:
                return word
        return None

    def get_candidates(self, word):
        candidates_lst = [] # First cell: one edit candidates, Second cell: two edits candidates
        one_edit_candidates = self.get_edits_by_one(word)
        two_edit_candidates = self.get_edits_by_two(one_edit_candidates)

        candidates_lst.append(one_edit_candidates)
        candidates_lst.append(two_edit_candidates)
        return candidates_lst

    def get_edits_by_one(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        insertion = self.get_insertion(splits, letters)
        transposition = self.get_transposition(splits, letters)
        substitution = self.get_substitution(splits, letters)
        deletion = self.get_deletion(splits, letters)

        deletion = self.remove_unknown_words(deletion)
        transposition = self.remove_unknown_words(transposition)
        substitution = self.remove_unknown_words(substitution)
        insertion = self.remove_unknown_words(insertion)

        edits_dict = {'deletion':set(deletion), 'transposition':set(transposition),
                      'substitution':set(substitution), 'insertion':set(insertion)}
        return edits_dict

    def get_edits_by_two(self, one_edit_candidates):
        error_types = one_edit_candidates.keys()
        two_edit_candidates = {f'{error_one}+{error_two}': set() for error_one in error_types for error_two in error_types}

        for error_one, candidate_set_one in one_edit_candidates.items():
            for tpl_one in candidate_set_one:
                candidate_one = tpl_one[0]
                two_letters_one = tpl_one[1]
                second_edits = self.get_edits_by_one(candidate_one)
                for error_two, candidate_set_two in second_edits.items():
                    for tpl_two in candidate_set_two:
                        candidate_two = tpl_two[0]
                        two_letters_two = tpl_two[1]
                        two_edit_candidates[f'{error_one}+{error_two}'].add((candidate_two, f'{two_letters_one}+{two_letters_two}'))
        return two_edit_candidates

    def get_deletion(self, splits, letters):
        deletion = []
        for L, R in splits:
            for c in letters:
                if L != '':
                    deletion.append((L + c + R, L[-1] + c))
                else:
                    deletion.append((L + c + R, '#' + c))
        return deletion

    def get_insertion(self, splits, letters):
        insertion = []
        for L, R in splits:
            if R:
                if L != '':
                    insertion.append((L + R[1:], L[-1] + R[0]))
                else:
                    insertion.append((L + R[1:], '#' + R[0]))
        return insertion

    def get_transposition(self, splits, letters):
        transposition = []
        for L, R in splits:
            if len(R) > 1:
                if R[0] == R[1]: continue
                transposition.append((L + R[1] + R[0] + R[2:], R[1] + R[0]))
        return transposition

    def get_substitution(self, splits, letters):
        substitution = []
        for L, R in splits:
            if R:
                for c in letters:
                    if c == R[0]: continue
                    substitution.append((L + c + R[1:], c + R[0]))
        return substitution

    def remove_unknown_words(self, lst):
        WORDS = self.lm.WORDS
        return [tpl for tpl in lst if tpl[0] in WORDS]


    class Language_Model:

        def __init__(self, n=3, chars = False):
            self.n = n
            self.chars = chars
            self.model_dict = None # key: N gram, value: how much appeared
            # For smoothing computition
            self.model_trimmed_dict = {} # key: N-1 gram, value: how much appeared
            # For generating text from model
            self.model_context_dict = {} # key N-1 gram, value: list of N grams that N-1 gram is their prefix
            # For spelling correction
            self.WORDS = None

        def build_model(self, text):
            if self.model_dict is None: self.model_dict = {}
            if not self.chars:
                str_parts = text.split()
            else:
                str_parts = [char for char in text if char != ' '] #Check for correctess !!!

            self.WORDS = self.build_word_vocabulary(str_parts)

            dict_lst = [self.model_dict, self.model_trimmed_dict]
            bound_lst = [len(str_parts) - self.n + 1, len(str_parts) - self.n + 2]

            for dct, bound, N in zip(dict_lst, bound_lst, [self.n, self.n - 1]):
                for i in range(bound):  # N-grams
                    curr_gram = ' '.join(str_parts[i:i+N])
                    if curr_gram in dct:
                        dct[curr_gram] += 1
                    else:
                        dct[curr_gram] = 1
                    if dct is self.model_dict:
                        trimmed_curr_gram = curr_gram[0:curr_gram.rindex(' ')]
                        trimmed_part = curr_gram[curr_gram.rindex(' ')+1:]
                        if trimmed_curr_gram in self.model_context_dict:
                            self.model_context_dict[trimmed_curr_gram].append(trimmed_part)
                        else:
                            gram_lst = [trimmed_part]
                            self.model_context_dict[trimmed_curr_gram]=gram_lst


        def get_model_dictionary(self):
            return self.model_dict

        def get_model_window_size(self):
            return self.n

        def generate(self, context=None, n=20):
            # What to do if the context is not in the dictionary? !!!
            if context is not None and len(context.split()) >= n:
                return (context.split())[0:n]
            if context is None:
                # if sampling is by uniform
                # context = self.choose_sample('uniform')

                # if sampling is by choise
                context = self.choose_sample('choise') # Check for right sampling !!!

            generated_text = context.split()
            while len(generated_text) < n:
                last_context = generated_text[-(self.n-1):]
                try:
                    last_context_list = self.model_context_dict[' '.join(last_context)]
                    random_number = random.randint(0, len(last_context_list)-1) # Check for right sampling !!!
                    generated_text.append(last_context_list[random_number])
                except KeyError: break # Stops when context in not in the text
            return ' '.join(generated_text)

        def evaluate(self, text):
            str_parts = text.split()
            bound = len(str_parts) - self.n + 1
            N = self.n
            prob = 1
            for i in range(bound):
                curr_gram = ' '.join(str_parts[i:i+N])
                trimmed_curr_gram = curr_gram[0:curr_gram.rindex(' ')]
                if curr_gram in self.model_dict and trimmed_curr_gram in self.model_trimmed_dict:
                    prob *= self.model_dict[curr_gram] / self.model_trimmed_dict[trimmed_curr_gram]
                else:
                    prob *= self.smooth(curr_gram)
            return math.log(prob, 10)


        def smooth(self, ngram):
            trimmed_ngram = ngram[0:ngram.rindex(' ')]
            V = len(self.model_trimmed_dict)
            upper_c = 0 if ngram not in self.model_dict else self.model_dict[ngram]
            lower_c = 0 if trimmed_ngram not in self.model_trimmed_dict else self.model_trimmed_dict[trimmed_ngram]
            return (upper_c+1) / (lower_c+V)

        # My methods
        def choose_sample(self, how):
            if how == 'uniform':
                context_keys = list(self.model_context_dict.keys())
                random_number = random.randint(0,len(context_keys) - 1)
                return context_keys[random_number]
            elif how == 'choise':
                context_keys, weights = [], []
                for key, value in self.model_context_dict.items():
                    context_keys.append(key)
                    weights.append(len(value))
                return (random.choices(context_keys, weights, k=1))[0]
        def build_word_vocabulary(self, str_parts):
            word_set = set()
            for word in str_parts:
                word_set.add(word)
            return word_set



def normalize_text(text):
    lowered_text = text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
    set_punctuations.add('\n')
    normalized_text = ''
    for index in range(len(lowered_text)):
        if lowered_text[index] not in set_punctuations:
            normalized_text += lowered_text[index]
        else:
            try:
                if lowered_text[index+1] != ' ' and lowered_text[index-1] != ' ':
                    normalized_text += ' '
            except IndexError:
                continue
    return normalized_text


def who_am_i():
    return {'name':'Maxim Zhivodrov', 'id':'317649606', 'email':'maximzh@post.bgu.ac.il'}


#---------------------------- Tests ----------------------------#
s_c = Spell_Checker()
l_m = s_c.Language_Model(n = 3)
s_c.add_language_model(l_m)
#--- my tests ---#
# l_m.build_model('Maxim is student and is student')
# print(l_m.model_dict)
# print(l_m.model_trimmed_dict)
# print(l_m.model_context_dict)
# print(l_m.generate('is student',n = 4))

#--- big.txt tests ---#
# print('#--- big.txt ---#')
# big = open('big.txt', 'r').read()
# start = datetime.datetime.now()
# big_normlized = normalize_text(big)
# end = datetime.datetime.now()
# print(f'Normlization took:   {end - start}')
# start = datetime.datetime.now()
# l_m.build_model(big_normlized)
# end = datetime.datetime.now()
# print(f'Building the model took:   {end - start}')
# print()
# print()
# candidate_lst = s_c.get_candidates('appple')
# for can in candidate_lst:
#     for k,v in can.items():
#         if len(v) != 0:
#             print(f'Error type: {k}')
#             print(f'Candidates: {v}')
#             print()
#             print()

#--- corpus.data tests ---#
print('#--- corpus.data ---#')
print()
corpus = open('corpus.data', 'r').read()
corpus = ' '.join(corpus.split('<s>'))
start = datetime.datetime.now()
corpus_normlized = normalize_text(corpus)
end = datetime.datetime.now()
print(f'Normlization took:   {end - start}')
start = datetime.datetime.now()
l_m.build_model(corpus_normlized)
end = datetime.datetime.now()
print(f'Building the model took:   {end - start}')
print()
print()
candidate_lst = s_c.get_candidates('appple')
for can in candidate_lst:
    for k,v in can.items():
        if len(v) != 0:
            print(f'Error type: {k}')
            print(f'Candidates: {v}')
            print()
            print()

