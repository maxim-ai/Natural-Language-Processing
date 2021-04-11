import math
import random
import datetime


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
        pass

    def evaluate(self, text):
        return self.lm.evaluate(text)

    class Language_Model:

        def __init__(self, n=3, chars = False):
            self.n = n
            self.chars = chars
            self.model_dict = None # key: N gram, value: how much appeared
            self.model_trimmed_dict = {} # key: N-1 gram, value: how much appeared
            self.model_context_dict = {} # key N-1 gram, value: list of N grams that N-1 gram is their prefix

        def build_model(self, text):
            if self.model_dict is None: self.model_dict = {}
            if not self.chars:
                str_parts = text.split()
            else:
                str_parts = [char for char in text if char != ' '] #Check for correctess !!!

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



def normalize_text(text):
    lowered_text = text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
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
# big = open('big.txt', 'r').read()
# start = datetime.datetime.now()
# big_normlized = normalize_text(big)
# end = datetime.datetime.now()
# print(f'Normlization took:   {end - start}')
# start = datetime.datetime.now()
# l_m.build_model(big_normlized)
# end = datetime.datetime.now()
# print(f'Building the model took:   {end - start}')

#--- corpus.data tests ---#
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

