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



def main():
    start = datetime.now()
    tweets_df_train=read_tsv('trump_train.tsv',['tweet_id','user_handle','tweet_text','time_stamp','device'])
    tweets_df_test = read_tsv('trump_test.tsv',['user_handle','tweet_text','time_stamp'])
    vectorizer = CountVectorizer(stop_words= 'english',lowercase=True)


    tweets_class_dict_train = separte_tweets(tweets_df_train)
    train_X = vectorizer.fit_transform(list(tweets_class_dict_train.keys())).toarray()
    train_Y = list(tweets_class_dict_train.values())

    tweets_class_dict_test = separte_tweets(tweets_df_test)
    test_X = vectorizer.transform(list(tweets_class_dict_test.keys())).toarray()
    test_Y = list(tweets_class_dict_test.values())

    print('\n--- Logistic regression model ---')
    LogReg_model = LogisticRegression()
    LogReg_model.fit(train_X, train_Y)
    predictions = LogReg_model.predict(test_X)
    print(f'\n{predictions}')
    print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    print(f'Time took: {datetime.now()-start}')

    start = datetime.now()
    print('\n\n--- SVC model ---')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    print(f'\n{predictions}')
    print(f'\nAccuracy: {accuracy_score(test_Y, predictions)}')
    print(f'Time took: {datetime.now() - start}')

def separte_tweets(tweets_df):
    tweet_class_dict = {}
    stop_words = set(stopwords.words('english'))
    for index, row in tweets_df.iterrows():
        if row['user_handle'] == 'realDonaldTrump':
            tweet_class_dict[preprocess(row['tweet_text'], stop_words)] = 0
            continue
        if row['user_handle'] == 'PressSec':
            tweet_class_dict[preprocess(row['tweet_text'], stop_words)] = 1
            continue
        if row['user_handle'] == 'POTUS':
            time_tweeted = datetime.strptime(row['time_stamp'], '%Y-%m-%d %H:%M:%S').date()
            trump_start, trump_end = [get_date(d) for d in ['2017-01-20','2021-01-20']]
            if trump_start <= time_tweeted <= trump_end: tweet_class_dict[preprocess(row['tweet_text'], stop_words)] = 0
            else: tweet_class_dict[preprocess(row['tweet_text'], stop_words)] = 1
            continue
        # Should consider to use the device

    return tweet_class_dict

def get_date(d):
    return datetime.strptime(d, '%Y-%m-%d').date()

def read_tsv(file_name, headers):
    import csv
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

if __name__ == '__main__':
    main()