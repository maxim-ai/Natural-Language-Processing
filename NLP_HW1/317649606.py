import math
import random
import datetime


class Spell_Checker:

    def __init__(self, lm = None):
        self.lm = lm
        self.et = None
        self.normalization_dict = {'deletion':{}, 'insertion':{}, 'substitution':{}, 'transposition':{}}

    def add_language_model(self, lm):
        self.lm = lm

    def build_model(self, text, n):
        pass

    def add_error_tables(self, error_tables):
        self.et = error_tables

    def spell_check(self, text, alpha):
        # Should input text be normalized in the method or given normalized? !!!
        str_parts = text.split()
        wrong_word = self.check_if_has_wrong_word(str_parts)
        if wrong_word is not None:
            if len(str_parts) < self.lm.n:
                fixed_word = self.simple_noisy_chanel(wrong_word)[0]
                return ' '.join([part if part!= wrong_word else fixed_word for part in str_parts])
            else:
                fixed_word = self.context_noisy_chanel(str_parts,wrong_word)[0]
                return ' '.join([part if part != wrong_word else fixed_word for part in str_parts])
        else:
            if len(str_parts) < self.lm.n:
                pass
            else:
                fixed_word, wrong_word = self.context_real_words_noisy_chanel(str_parts, alpha)
                return (' '.join(str_parts)).replace(wrong_word, fixed_word)

    def evaluate(self, text):
        return self.lm.evaluate(text)

    # region My method spelling checker


    # region Get edit candidates
    def get_candidates(self, word):
        one_edit_candidates = self.get_edits_by_one(word=word)
        two_edit_candidates = self.get_edits_by_two(self.get_edits_by_one(word = word, two_edits_first_round=True), word)
        return one_edit_candidates, two_edit_candidates

    def get_edits_by_one(self, word, two_edits_first_round = False, original_word = None):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        insertion = self.get_insertion(splits, letters)
        transposition = self.get_transposition(splits, letters)
        substitution = self.get_substitution(splits, letters)
        deletion = self.get_deletion(splits, letters)

        if not two_edits_first_round: # For not removing if two edits needed
            if original_word is not None: word = original_word
            deletion = self.remove_redundant_words(deletion, word)
            transposition = self.remove_redundant_words(transposition, word)
            substitution = self.remove_redundant_words(substitution, word)
            insertion = self.remove_redundant_words(insertion, word)


        edits_dict = {'deletion':set(deletion),
                      'transposition':set(transposition),
                      'substitution':set(substitution),
                      'insertion':set(insertion)}
        return edits_dict

    def get_edits_by_two(self, one_edit_candidates, original_word):
        error_types = one_edit_candidates.keys()
        two_edit_candidates = {f'{error_one}+{error_two}': set() for error_one in error_types for error_two in error_types}

        for error_one, candidate_set_one in one_edit_candidates.items():
            for tpl_one in candidate_set_one:
                candidate_one = tpl_one[0]
                two_letters_one = tpl_one[1]
                second_edits = self.get_edits_by_one(candidate_one, original_word = original_word)
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
    # endregion

    # region Normzalization methods
    def deletion_normalization(self, string):
        if string in self.normalization_dict['deletion']:
            return self.normalization_dict['deletion'][string]
        if string[0] != '#':
            count_appearences = len([1 for k in self.lm.WORDS.keys() if string in k])
        else:
            count_appearences = len([1 for k in self.lm.WORDS.keys() if k[0] == string[1]])
        self.normalization_dict['deletion'][string] = count_appearences
        return count_appearences

    def insertion_normalization(self, string):
        if string in self.normalization_dict['insertion']:
            return self.normalization_dict['insertion'][string]
        if string[0] != '#':
            count_appearences = len([1 for k in self.lm.WORDS.keys() if string[0] in k])
        else:
            count_appearences = len(self.lm.WORDS)
        self.normalization_dict['insertion'][string] = count_appearences
        return count_appearences

    def substitution_normalization(self, string):
        if string in self.normalization_dict['substitution']:
            return self.normalization_dict['substitution'][string]
        count_appearences = len([1 for k in self.lm.WORDS.keys() if string[1] in k])
        self.normalization_dict['substitution'][string] = count_appearences
        return count_appearences

    def transposition_normalization(self, string):
        if string in self.normalization_dict['transposition']:
            return self.normalization_dict['transposition'][string]
        count_appearences = len([1 for k in self.lm.WORDS.keys() if string in k])
        self.normalization_dict['transposition'][string] = count_appearences
        return count_appearences
    # endregion

    def simple_noisy_chanel(self, word):
        candidates_mistake_probs = self.calculate_mistake_prob(word)
        candidates_chanel_probs = []
        for candidate_mistake_tpl in candidates_mistake_probs:
            candidate_word = candidate_mistake_tpl[0]
            mistake_prob = candidate_mistake_tpl[1]
            candidates_chanel_probs.append((candidate_word,mistake_prob*self.calculate_prior_prob(candidate_word)))

        max_prob_tpl = self.get_tuple_with_max_values(candidates_chanel_probs)
        return max_prob_tpl

    def context_noisy_chanel(self,str_parts ,word):
        all_grams_containing_word = self.get_all_grams_contains_the_word(str_parts, word)
        candidates_mistakes_probs  = self.calculate_mistake_prob(word)
        candidates_chanel_probs = []

        for candidate_mistake_tpl in candidates_mistakes_probs:
            candidate_word = candidate_mistake_tpl[0]
            mistake_prob = candidate_mistake_tpl[1]
            if mistake_prob == 0: continue
            all_grams_containing_candidate_word = [gram.replace(word, candidate_word) for gram in all_grams_containing_word]
            gram_probabilty = 1
            for gram in all_grams_containing_candidate_word:
                gram_probabilty *= math.pow(10, self.evaluate(gram))
            candidates_chanel_probs.append((candidate_word, mistake_prob*gram_probabilty))

        max_prob_tpl = self.get_tuple_with_max_values(candidates_chanel_probs)
        return max_prob_tpl

    def context_real_words_noisy_chanel(self, str_parts, alpha):
        candidate_for_real_mistake = {} # Key: Tuple of (candidate_word, probability), Value: Original word
        for word in str_parts:
            best_context_candidate_tpl = self.context_noisy_chanel(str_parts, word)
            all_grams_containing_word = self.get_all_grams_contains_the_word(str_parts, word)
            gram_probability = 1
            for gram in all_grams_containing_word:
                gram_probability *= math.pow(10, self.evaluate(gram))

            if alpha * gram_probability > (1 - alpha) * best_context_candidate_tpl[1]:
                candidate_for_real_mistake[(word, alpha * gram_probability)] = word
            else:
                candidate_for_real_mistake[best_context_candidate_tpl] = word

        max_prob_tpl = self.get_tuple_with_max_values(list(candidate_for_real_mistake.keys()))
        return max_prob_tpl[0], candidate_for_real_mistake[max_prob_tpl]

    def calculate_prior_prob(self, word):
        WORDS = self.lm.WORDS
        N = sum(WORDS.values())
        return (WORDS[word]) / N

    def calculate_mistake_prob(self, word):
        candidates_mistake_probs = [] # Tuples of (candidate_word, mistake_probabilty)
        candidates = self.get_candidates(word)
        one_edit_candidates = candidates[0]
        two_edit_candidates = candidates[1]
        norm_methods = {'deletion':self.deletion_normalization, 'insertion': self.insertion_normalization,
                        'substitution':self.substitution_normalization, 'transposition':self.transposition_normalization}

        for error_type, candidates_letters_tpls in one_edit_candidates.items():
            for can_let_tpl in candidates_letters_tpls:
                candidate_word = can_let_tpl[0]
                letters_modified = can_let_tpl[1]
                try:
                    error_model_prob = self.et[error_type][letters_modified]
                    if error_model_prob == 0: continue
                    mistake_prob = error_model_prob / (norm_methods[error_type](letters_modified))
                except ZeroDivisionError:
                    mistake_prob = 0
                candidates_mistake_probs.append((candidate_word, mistake_prob))

        for error_type, candidates_letters_tpls in two_edit_candidates.items():
            first_error, second_error = error_type.split('+')[0], error_type.split('+')[1]
            for can_let_tpl in candidates_letters_tpls:
                candidate_word = can_let_tpl[0]
                letters_modified = can_let_tpl[1]
                first_letters_modified, second_letters_modified = letters_modified.split('+')[0], letters_modified.split('+')[1]
                try:
                    error_model_prob_one = self.et[first_error][first_letters_modified]
                    error_model_prob_two = self.et[second_error][second_letters_modified]
                    if error_model_prob_one == 0 or error_model_prob_two == 0: continue
                    first_mistake_prob = error_model_prob_one / (norm_methods[first_error](first_letters_modified))
                    second_mistake_prob = error_model_prob_two / (norm_methods[second_error](second_letters_modified))
                    mistake_prob = first_mistake_prob * second_mistake_prob
                except ZeroDivisionError:
                    mistake_prob = 0
                candidates_mistake_probs.append((candidate_word, mistake_prob))

        return candidates_mistake_probs

    # region Helpful methods
    def get_all_grams_contains_the_word(self, str_parts, word):
        all_grams = set()
        bound = len(str_parts) - self.lm.n + 1
        N = self.lm.n
        for i in range(bound):
            curr_gram = ' '.join(str_parts[i:i + N])
            all_grams.add(curr_gram)
        return [gram for gram in all_grams if word in gram]

    def get_tuple_with_max_values(self, lst):
        max_value_tpl = (0, 0)
        for tpl in lst:
            if tpl[1] > max_value_tpl[1]:
                max_value_tpl = tpl
        return max_value_tpl

    def check_if_has_wrong_word(self, str_parts):
        WORDS = self.lm.WORDS
        for word in str_parts:
            if word not in WORDS:
                return word
        return None

    def remove_redundant_words(self, lst, original_word):
        WORDS = self.lm.WORDS
        return [tpl for tpl in lst if tpl[0] in WORDS and tpl[0] != original_word]

    def remove_original_word(self, lst, original_word):
        return [tpl for tpl in lst if tpl[0] != original_word]
    # endregion


    # endregion


    # region Language model inner class
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
            word_dict = {}
            for word in str_parts:
                if word in word_dict:
                    word_dict[word] +=1
                else:
                    word_dict[word] = 1
            return word_dict
    # endregion



# region Normalization and WhoAmI methods
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
# endregion


#---------------------------- Tests ----------------------------#
s_c = Spell_Checker()
l_m = s_c.Language_Model(n = 3)
s_c.add_language_model(l_m)
from spelling_confusion_matrices import error_tables
s_c.add_error_tables(error_tables)
#--- my tests ---#
# l_m.build_model('Maxim is student and is student')
# print(l_m.model_dict)
# print(l_m.model_trimmed_dict)
# print(l_m.model_context_dict)
# print(l_m.generate('is student',n = 4))

#--- big.txt tests ---#
print('#--- big.txt ---#')
big = open('big.txt', 'r').read()
start = datetime.datetime.now()
big_normlized = normalize_text(big)
end = datetime.datetime.now()
print(f'Normlization took:   {end - start}')
start = datetime.datetime.now()
l_m.build_model(big_normlized)
end = datetime.datetime.now()
print(f'Building the model took:   {end - start}')
print()
print(s_c.spell_check('two of them apples', 0.95))
print(s_c.spell_check('he got a pretty good karacter',0.95))
print(s_c.spell_check('i acress the room',0.95))
print(s_c.spell_check('the famous acress',0.95))
print(s_c.spell_check('acress put the apple on the table',0.95))
print(s_c.spell_check('i eat appple every day',0.95))


#--- corpus.data tests ---#
# print('#--- corpus.data ---#')
# print()
# corpus = open('corpus.data', 'r').read()
# corpus = ' '.join(corpus.split('<s>'))
# start = datetime.datetime.now()
# corpus_normlized = normalize_text(corpus)
# end = datetime.datetime.now()
# print(f'Normlization took:   {end - start}')
# start = datetime.datetime.now()
# l_m.build_model(corpus_normlized)
# end = datetime.datetime.now()
# print(f'Building the model took:   {end - start}')
# print()
# print(s_c.spell_check('he got a pretty good karacter',0.95))
# print(s_c.spell_check('i acress the room',0.95))
# print(s_c.spell_check('acress put the apple on the table',0.95))
# print(s_c.spell_check('i eat appple every day',0.95))
# start = datetime.datetime.now()
# print(s_c.spell_check('two of the kings', 0.95))
# end = datetime.datetime.now()
# print(f'Correction of the sentence took: {end - start}')


