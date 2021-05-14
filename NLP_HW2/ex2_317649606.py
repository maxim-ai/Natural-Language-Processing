import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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

    tweet_class_list = separte_tweets(tweets_df_train) #Tuple: (pp tweet, class, org tweet)

    #For training
    train_X_splitted, test_X_splitted, train_Y_splitted, test_Y_splitted = split_train_test(tweet_class_list)


    #For testing
    # train_X_splitted = [(tpl[0],tpl[2]) for tpl in tweet_class_list]
    # test_X_splitted = get_test_data()
    # train_Y_splitted = [tpl[1] for tpl in tweet_class_list]
    # test_Y_splitted = []


    original_train_tweets = [tpl[1] for tpl in train_X_splitted]
    original_test_tweets = [tpl[1] for tpl in test_X_splitted]
    tweets_class_list = [(tpl[0], tpl[1]) for tpl in tweet_class_list]
    train_X_splitted = [tpl[0] for tpl in train_X_splitted]
    test_X_splitted = [tpl[0] for tpl in test_X_splitted]






    # Without vectorizer !!!
    # vocabulary = set(token for tweet in train_X_splitted for token in tweet.split())
    # train_X = my_fit_transform(train_X_splitted, vocabulary)
    # train_Y = train_Y_splitted
    # test_X = my_transform(test_X_splitted, vocabulary)
    # test_Y = test_Y_splitted


    vectorizer = CountVectorizer(stop_words='english', lowercase=True)

    train_X = form_features(vectorizer.fit_transform(train_X_splitted).toarray(), original_train_tweets)
    train_Y = train_Y_splitted
    test_X = form_features(vectorizer.transform(test_X_splitted).toarray(), original_test_tweets)
    test_Y = test_Y_splitted



    # start = datetime.now()
    # print('\n------------------------------ Logistic regression model ------------------------------')
    # LogReg_model = LogisticRegression(max_iter=400, multi_class='ovr')
    # LogReg_model.fit(train_X, train_Y)
    # predictions = LogReg_model.predict(test_X)
    # # write_test_predictions(predictions)
    # print(f'\n{predictions}')
    # print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    # print(f'Time took: {datetime.now()-start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_list, LogReg_model, 5,vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now()-start}')



    # start = datetime.now()
    # print('\n\n\n------------------------------ SVC linear model ------------------------------')
    # SVClin_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel = 'linear'))
    # SVClin_model.fit(train_X, train_Y)
    # predictions = SVClin_model.predict(test_X)
    # print(f'\n{predictions}')
    # print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    # print(f'Time took: {datetime.now() - start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_list, SVClin_model, 5, vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now() - start}')

    # start = datetime.now()
    # print('\n\n\n------------------------------ SVC non_linear model ------------------------------')
    # SVCnonlin_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='sigmoid', C=10))
    # SVCnonlin_model.fit(train_X, train_Y)
    # predictions = SVCnonlin_model.predict(test_X)
    # print(f'\n{predictions}')
    # print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    # print(f'Time took: {datetime.now() - start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_list, SVCnonlin_model, 5, vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now() - start}')


    # start = datetime.now()
    # print('\n\n\n------------------------------ FFNN model ------------------------------')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tweet_dict = make_dict([[token for token in tpl[0].split()] for tpl in tweets_class_list], False)
    #
    # input_dim = len(tweet_dict)
    # hidden_dim = 500
    # output_dim = 2
    # num_epochs = 20
    #
    # ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    # ff_nn_bow_model.to(device)
    #
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.001)
    #
    # for epoch in range(num_epochs):
    #     for index, tweet in enumerate(train_X_splitted):
    #         optimizer.zero_grad()
    #         bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
    #         probs = ff_nn_bow_model(bow_vec)
    #         target = make_target(train_Y_splitted[index], device)
    #         loss = loss_function(probs, target)
    #         loss.backward()
    #         optimizer.step()
    #     print(f'{epoch + 1} epoch completed')
    #
    # bow_ff_nn_predictions = []
    # original_lables_ff_bow = []
    # with torch.no_grad():
    #     for index, tweet in enumerate(test_X_splitted):
    #         bow_vec = make_bow_vector(tweet_dict, tweet.split(), device)
    #         probs = ff_nn_bow_model(bow_vec)
    #         bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
    #         original_lables_ff_bow.append(make_target(test_Y_splitted[index], device).cpu().numpy()[0])
    #
    # print(classification_report(original_lables_ff_bow,bow_ff_nn_predictions))
    # print(f'\n\nAccuracy: {accuracy_score(original_lables_ff_bow,bow_ff_nn_predictions)}')
    # print(f'Time took: {datetime.now() - start}')


    start = datetime.now()
    print('\n------------------------------ Naive bayes model ------------------------------')
    NaiveBayes_model = MultinomialNB(fit_prior=False)
    NaiveBayes_model.fit(train_X, train_Y)
    predictions = NaiveBayes_model.predict(test_X)
    print(f'\n{predictions}')
    print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    print(f'Time took: {datetime.now()-start}')
    # start = datetime.now()
    # cv_scores = use_cross_validation(tweets_class_list, NaiveBayes_model, 5,vectorizer)
    # print(f'\nAccuracy Cross-validation: {np.average(cv_scores)}')
    # print(f'Time took: {datetime.now()-start}')


def separte_tweets(tweets_df):
    """
    Make a list of tuples contains pp tweet, class and original tweet
    Args:
        tweets_df (dataframe): dataframe after reading the tsv file
    Returns:
            (list): list of tuples
    """
    stop_words = set(stopwords.words('english'))
    tweet_class_list = []
    for index, row in tweets_df.iterrows():
        pp_tweet = preprocess(row['tweet_text'], stop_words)
        original_tweet = row['tweet_text']
        if row['user_handle'] == 'realDonaldTrump':
            device, tweet_date = row['device'], get_date(row['time_stamp'])
            change_device_date = get_date('2017-04-01')
            if device == 'android':
                if tweet_date < change_device_date:
                    tweet_class_list.append((pp_tweet, 0, original_tweet))
                else:
                    tweet_class_list.append((pp_tweet, 1, original_tweet))
            elif device == 'iphone':
                if tweet_date > change_device_date:
                    tweet_class_list.append((pp_tweet, 0, original_tweet))
                else:
                    tweet_class_list.append((pp_tweet, 1, original_tweet))
            else:
                tweet_class_list.append((pp_tweet, 1, original_tweet))
        elif row['user_handle'] == 'PressSec':
            tweet_class_list.append((pp_tweet, 1, original_tweet))
        elif row['user_handle'] == 'POTUS':
            time_tweeted = get_date(row['time_stamp'])
            trump_start, trump_end = [get_date(d) for d in ['2017-01-20','2021-01-20']]
            if trump_start <= time_tweeted <= trump_end:
                tweet_class_list.append((pp_tweet, 0, original_tweet))
            else:
                tweet_class_list.append((pp_tweet, 1, original_tweet))
    return tweet_class_list


def get_date(d):
    """
    Constructs a date object from date string
    Args:
        d (str): a date string
    Returns:
            (date): date object
    """
    if ':' in d:
        return datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date()
    else:
        return datetime.strptime(d, '%Y-%m-%d').date()


def my_fit_transform(train_X_splitted, tokens):
    """
    Make 2D array of tweets and token features
    Args:
        train_X_splitted (list): a list of tweets
        tokens (set): vocabulary of tokens
    Returns:
            (np-array): 2D array
    """
    tweet_index = {tweet:index for index,tweet in enumerate(train_X_splitted)}
    token_index = {token:index for index,token in enumerate(tokens)}
    feature_arr = np.array([[0 for token in tokens] for tweet in train_X_splitted])
    for tweet in train_X_splitted:
        for token in tweet.split():
            feature_arr[tweet_index[tweet]][token_index[token]] += 1
    return feature_arr

def my_transform(test_X_splitted, tokens):
    """
    Make 2D array of tweets and token features
    Args:
        test_X_splitted (list): a list of tweets
        tokens (set): vocabulary of tokens
    Returns:
            (np-array): 2D array
    """
    tweet_index = {tweet: index for index, tweet in enumerate(test_X_splitted)}
    token_index = {token: index for index, token in enumerate(tokens)}
    feature_arr = np.array([[0 for token in tokens] for tweet in test_X_splitted])
    for tweet in test_X_splitted:
        for token in tweet.split():
            try:
                feature_arr[tweet_index[tweet]][token_index[token]] += 1
            except KeyError:
                continue
    return feature_arr



def get_length_feature(tweets):
    """
    Makes length feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of length feature
    """
    return np.array([[len(tweet.split())] for tweet in tweets])

def get_hashtag_feature(tweets):
    """
    Makes hashtag feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of hashtag feature
    """
    return np.array([[tweet.count('#')] for tweet in tweets])

def get_capitalletter_feature(tweets):
    """
    Makes capital letters feature vector from the tweets
    Args:
        tweets (list): list of tweets
    Returns:
            (np-array): 1D array of length feature
    """
    return np.array([[len([1 for token in tweet.split() if token[0].isupper()])] for tweet in tweets])

def form_features(twoD_array, original_tweets):
    """
    Makes 2D array from tweets with all features
    Args:
         twoD_array (np-array): 2D array of token features
         original_tweets (list): list of original tweets
    Returns:
            (np-array): 2D array of all features
    """
    features_arrays_train = [get_length_feature(original_tweets),
                             get_hashtag_feature(original_tweets),
                             get_capitalletter_feature(original_tweets)]
    for features in features_arrays_train:
        twoD_array = np.hstack((twoD_array, features))
    return twoD_array

def read_tsv(file_name, headers):
    """
    Reads tsv file
    Args:
        file_name (str): the name of the file
        headers (list): the headers of the table
    Returns:
            (dataframe): dataframe containg the tsv file
    """
    tsvfile = open(file_name,'r')
    tsvreader = csv.reader(tsvfile, delimiter = '\n')
    tweet_list = [line[0] for line in tsvreader]
    tweet_dict = {header:[] for header in headers}
    for tweet in tweet_list:
        splitted_tweet = tweet.split('\t')
        for part, header in zip(splitted_tweet,headers):
            tweet_dict[header].append(part)
    return pd.DataFrame(tweet_dict)


def preprocess(text, stop_words):
    """
    Preprocess input text
    Args:
        text (str): text to preprocess
        stop_words (list): list of stop-words
    Returns:
            (str): preprocessed text
    """
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
    """
    Splits data to train and test sets
    Args:
        XY_list (list): list of tuples containing the data
    Returns:
            (tuple): four list of XY train and test
    """
    return train_test_split([(tpl[0], tpl[2]) for tpl in XY_list], [tpl[1] for tpl in XY_list], test_size=0.2 ,shuffle=True)

def use_cross_validation(tweet_class_list, model, folds,vectorizer):
    """
    Applys cross validation on train and test set with specific model
    Args:
         tweet_class_list (list): list of tuples
         model (model): the model to use cross validation on
         folds (int): how much folds
         vectorizer (vectorizer): vectorizer object for feature exraction
    Returns:
            (list): list of acccuracies
    """
    return cross_val_score(model, vectorizer.fit_transform([tpl[0] for tpl in tweet_class_list]).toarray(),
                    [str(value) for value in [tpl[1] for tpl in tweet_class_list]],
                    cv=folds)

def get_test_data():
    """
    Reads the test data and applys preprocessing on it
    Returns:
            (list): list of tuples of pp tweet and original tweet
    """
    stop_words = set(stopwords.words('english'))
    tweets_df_test = read_tsv('trump_test.tsv',['user_handle','tweet_text','time_stamp'])
    tweet_list = []
    for index, row in tweets_df_test.iterrows():
        pp_tweet = preprocess(row['tweet_text'], stop_words)
        original_tweet = row['tweet_text']
        tweet_list.append((pp_tweet, original_tweet))
    return tweet_list

def write_test_predictions(predictions):
    """
    Writes test prediction to file
    Args:
         predictions (list): list of 0,1 predictions
    """
    result_string = ' '.join([str(pred) for pred in predictions])
    open('results','w').write(result_string)


def make_dict(token_list, padding=True):
    """
    Makes dictionary from the dataset
    Args:
        token_list (list): list of tokens of the dataset
        padding (bool): whether to apply padding
    Returns:
            (dict): dictionary of the dataset
    """
    if padding:
        tweet_dict = corpora.Dictionary([['pad']])
        tweet_dict.add_documents(token_list)
    else:
        tweet_dict = corpora.Dictionary(token_list)
    return tweet_dict

def make_bow_vector(tweet_dict, sentence, device):
    """
    Makes bow vector from sentence (tweet)
    Args:
        tweet_dict (dict): dictionary containing tweets
        sentence (str): the tweet
        device (device): cpu/gpu
    Returns:
            (list): a bow vector
    """
    vec = torch.zeros(len(tweet_dict), dtype=torch.float64, device=device)
    for word in sentence:
        vec[tweet_dict.token2id[word]] += 1
    return vec.view(1, -1).float()

def make_target(label, device):
    """
    Makes a target vector
    Args:
        label (int): the class of the tweet
        device (device): cpu/gpu
    Returns:
            (list): a target vector
    """
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