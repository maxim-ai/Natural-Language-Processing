import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import numpy as np
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from gensim import corpora

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from sklearn.metrics import classification_report


def main():

    tweets_df_train=read_tsv('trump_train.tsv',['tweet_id','user_handle','tweet_text','time_stamp','device'])
    vectorizer = CountVectorizer(stop_words= 'english',lowercase=True)

    tweets_class_list = separte_tweets(tweets_df_train)

    train_X_splitted, test_X_splitted, train_Y_splitted, test_Y_splitted = split_train_test(tweets_class_list)

    train_X = vectorizer.fit_transform(train_X_splitted).toarray()
    train_Y = train_Y_splitted

    test_X = vectorizer.transform(test_X_splitted).toarray()
    test_Y = test_Y_splitted


    start = datetime.now()
    print('\n------------------------------ Logistic regression model ------------------------------')
    LogReg_model = LogisticRegression()
    LogReg_model.fit(train_X, train_Y)
    predictions = LogReg_model.predict(test_X)
    print(f'\n{predictions}')
    print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    print(f'Time took: {datetime.now()-start}')
    start = datetime.now()
    cv_scores = use_cross_validation(tweets_class_list, LogReg_model, 5,vectorizer)
    print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    print(f'Time took: {datetime.now()-start}')



    # start = datetime.now()
    # print('\n\n\n------------------------------ SVC linear model ------------------------------')
    # SVClin_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel = 'linear'))
    # SVClin_model.fit(train_X, train_Y)
    # predictions = SVClin_model.predict(test_X)
    # print(f'\n{predictions}')
    # print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    # print(f'Time took: {datetime.now() - start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_dict, SVClin_model, 5, vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now() - start}')

    # start = datetime.now()
    # print('\n\n\n------------------------------ SVC non_linear model ------------------------------')
    # SVCnonlin_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='sigmoid'))
    # SVCnonlin_model.fit(train_X, train_Y)
    # predictions = SVCnonlin_model.predict(test_X)
    # print(f'\n{predictions}')
    # print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    # print(f'Time took: {datetime.now() - start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_dict, SVCnonlin_model, 5, vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now() - start}')


    start = datetime.now()
    print('\n\n\n------------------------------ FFNN model ------------------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tweet_dict = make_dict([[token for token in tpl[0].split()] for tpl in tweets_class_list], False)

    input_dim = len(tweet_dict)
    hidden_dim = 500
    output_dim = 2
    num_epochs = 20

    ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    ff_nn_bow_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for index, tweet in enumerate(train_X_splitted):
            optimizer.zero_grad()
            bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
            probs = ff_nn_bow_model(bow_vec)
            target = make_target(train_Y_splitted[index], device)
            loss = loss_function(probs, target)
            loss.backward()
            optimizer.step()
        print(f'{epoch + 1} epoch completed')

    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, tweet in enumerate(test_X_splitted):
            bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(test_Y_splitted[index], device).cpu().numpy()[0])

    print(classification_report(original_lables_ff_bow,bow_ff_nn_predictions))
    print(f'\n\nAccuracy: {accuracy_score(original_lables_ff_bow,bow_ff_nn_predictions)}')
    print(f'Time took: {datetime.now() - start}')


def separte_tweets(tweets_df):
    stop_words = set(stopwords.words('english'))
    # tweet_class_dict = {}
    tweet_class_list = []
    for index, row in tweets_df.iterrows():
        pp_tweet = preprocess(row['tweet_text'], stop_words)
        if row['user_handle'] == 'realDonaldTrump':
            device, tweet_date = row['device'], get_date(row['time_stamp'])
            change_device_date = get_date('2017-04-01')
            if device == 'android':
                if tweet_date < change_device_date:
                    tweet_class_list.append((pp_tweet, 0))
                    # tweet_class_dict[pp_tweet] = 0
                else:
                    tweet_class_list.append((pp_tweet, 1))
                    # tweet_class_dict[preprocess(row['tweet_text'],stop_words)] = 1
            elif device == 'iphone':
                if tweet_date > change_device_date:
                    tweet_class_list.append((pp_tweet, 0))
                    # tweet_class_dict[pp_tweet] = 0
                else:
                    tweet_class_list.append((pp_tweet, 1))
                    # tweet_class_dict[pp_tweet] = 1
            else:
                tweet_class_list.append((pp_tweet, 1))
                # tweet_class_dict[pp_tweet] = 1
        elif row['user_handle'] == 'PressSec':
            tweet_class_list.append((pp_tweet, 1))
            # tweet_class_dict[pp_tweet] = 1
        elif row['user_handle'] == 'POTUS':
            time_tweeted = get_date(row['time_stamp'])
            trump_start, trump_end = [get_date(d) for d in ['2017-01-20','2021-01-20']]
            if trump_start <= time_tweeted <= trump_end:
                tweet_class_list.append((pp_tweet, 0))
                # tweet_class_dict[pp_tweet] = 0
            else:
                tweet_class_list.append((pp_tweet, 1))
                # tweet_class_dict[pp_tweet] = 1
    return tweet_class_list

def get_date(d):
    if ':' in d:
        return datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date()
    else:
        return datetime.strptime(d, '%Y-%m-%d').date()

def read_tsv(file_name, headers):
    tsvfile = open(file_name,'r')
    tsvreader = csv.reader(tsvfile, delimiter = '\n')
    tweet_list = [line[0] for line in tsvreader]
    tweet_dict = {header:[] for header in headers}
    for tweet in tweet_list:
        splitted_tweet = tweet.split('\t')
        for part, header in zip(splitted_tweet,headers):
            tweet_dict[header].append(part)
    return pd.DataFrame(tweet_dict)

def trans_nparray(array):
    return [[elm] for elm in array]

def preprocess(text, stop_words):
    lowered_text = text.lower()
    set_punctuations = {char for char in '''!()-[]{};:'"\,<>./?@#$%^&*_~'''}
    set_punctuations.add('\n')
    normalized_text = ''
    for index in range(len(lowered_text)):
        try:
            if lowered_text[index] not in set_punctuations:
                normalized_text += lowered_text[index]
            else:
                try:
                    if lowered_text[index+1] != ' ' and lowered_text[index-1] != ' ':
                        normalized_text += ' '
                except IndexError:
                    continue
        except UnicodeDecodeError:
            continue

    normalized_text = ' '.join([part for part in normalized_text.split() if part not in stop_words])

    return normalized_text

def split_train_test(XY_list):
    return train_test_split([tpl[0] for tpl in XY_list], [tpl[1] for tpl in XY_list], test_size=0.2 ,shuffle=True)

def use_cross_validation(tweet_class_list, model, folds,vectorizer):
    return cross_val_score(model, vectorizer.fit_transform([tpl[0] for tpl in tweet_class_list]).toarray(),
                    [str(value) for value in [tpl[1] for tpl in tweet_class_list]],
                    cv=folds)

def make_dict(token_list, padding=True):
    if padding:
        tweet_dict = corpora.Dictionary([['pad']])
        tweet_dict.add_documents(token_list)
    else:
        tweet_dict = corpora.Dictionary(token_list)
    return tweet_dict

def make_bow_vector(tweet_dict, sentence, device):
    vec = torch.zeros(len(tweet_dict), dtype=torch.float64, device=device)
    for word in sentence:
        vec[tweet_dict.token2id[word]] += 1
    return vec.view(1, -1).float()

def make_target(label, device):
    if label == 0:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 1:
        return torch.tensor([1], dtype=torch.long, device=device)


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function 1: vocab_size --> 500
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # # Linear function 2: 500 --> 500
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # # Non-linearity 2
        # self.relu2 = nn.ReLU()

        # Linear function 2 (readout): 500 --> 3
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # # Linear function 2
        # out = self.fc2(out)
        # # Non-linearity 2
        # out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc2(out)

        return F.softmax(out, dim=1)



if __name__ == '__main__':
    main()