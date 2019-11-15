#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:22:00 2019

@author: lindaxju
"""

#%%
import pandas as pd
import numpy as np

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

import pickle
from datetime import datetime

#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#%%
##################################Choose Data##################################
df_orig = pd.read_csv('data/trump_tweets_19-05-11.csv')
#%%
df_orig.head()
#%%
df_orig.shape
#%%
df_full = df_orig.copy()
df_full['is_retweet_new'] = df_full.text.str.startswith('RT')
df_full['created_at_dt'] = pd.to_datetime(df_full['created_at'])
#%%
df_full.info()
#%%
df_full.shape
#%%
df_noRT = df_full[df_full.is_retweet_new==False]
df_noRT.head()
#%%
df_noRT.info()
#%%
df_noRT.shape
#%%
df_final = df_noRT.copy()
df_final.reset_index(inplace=True)
df_final.rename(columns={'index': 'index_orig'},inplace=True)
df_final.info()
#%%
Counter(df_final.is_retweet_new)
#%%
#################################Preprocessing#################################
#%%
# Drop unnecessary columns
df_final_preclean = df_final.copy()
drop_cols_list = ['source','created_at','retweet_count','favorite_count',
                  'is_retweet','id_str','is_retweet_new']
df_final_preclean.drop(drop_cols_list,axis=1,inplace=True)
df_final_preclean.head()
#%%
# Column to show those over character limit
df_final_preclean['preclean_len'] = [len(t) for t in df_final_preclean.text]
char_limit = 280
df_final_preclean[df_final_preclean.preclean_len>char_limit]
#%%
# cleans capital letters, punctuation, and numbers
def clean_tweet1(tweet):
    # decode HTML
    tweet_soup = BeautifulSoup(tweet, 'lxml')
    tweet_clean = tweet_soup.get_text()
    # remove URL links
    tweet_clean = re.sub('https?://[A-Za-z0-9./]+','',tweet_clean)
    # remove @, #, numbers, and punctuation
    tweet_clean = re.sub(r'[^a-zA-Z]+',' ', tweet_clean)
    # make lower case
    tweet_clean = tweet_clean.lower()
    tweet_clean = tweet_clean.strip()
    return tweet_clean
#%%
index_orig = []
clean_tweet_texts = []
for i in range(len(df_final_preclean.text)):
    index_orig.append(df_final_preclean.index_orig[i])
    clean_tweet_texts.append(clean_tweet1(df_final_preclean.text[i]))
#%%
print(clean_tweet_texts[:10])
print(df_final_preclean.shape)
print(len(clean_tweet_texts)) # Check same number of clean tweets as pre-clean tweets
#%%
df_clean1 = df_final_preclean.copy()
df_clean1.drop(['preclean_len'],axis=1,inplace=True)
df_clean1['text1_clean'] = clean_tweet_texts
df_clean1.head()
#%%
df_clean1.info()
#%%
df_clean1.shape
#%%
# cleans stemming
def clean_tweet2(tweet):
    store_stop_words = set(stopwords.words('english'))
    stemmer = LancasterStemmer()
    tweet_clean = ''
    for word in tweet.split():
        if word not in store_stop_words:
            tweet_clean += stemmer.stem(word) + ' '
    tweet_clean = tweet_clean.strip()
    return tweet_clean
#%%
index_orig = []
clean_tweet_texts = []
for i in range(len(df_clean1.text1_clean)):
    index_orig.append(df_clean1.index_orig[i])
    clean_tweet_texts.append(clean_tweet2(df_clean1.text1_clean[i]))
#%%
print(clean_tweet_texts[:10])
print(df_clean1.shape)
print(len(clean_tweet_texts)) # Check same number of clean tweets as pre-clean tweets
#%%
df_clean2 = df_clean1.copy()
df_clean2['text2_stopwords_stemmed'] = clean_tweet_texts
df_clean2.head()
#%%
analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return score
#%%
df_clean3 = df_clean2.copy()
list_neg = []
list_neu = []
list_pos = []
list_compound = []
for i in range(len(df_clean2)):
    tweet = df_clean3['text1_clean'][i]
    score = sentiment_analyzer_scores(tweet)
    list_neg.append(score['neg'])
    list_neu.append(score['neu'])
    list_pos.append(score['pos'])
    list_compound.append(score['compound'])
df_clean3['neg'] = list_neg
df_clean3['neu'] = list_neu
df_clean3['pos'] = list_pos
df_clean3['compound'] = list_compound
df_clean3['sentiment'] = [compound >= 0 for compound in list_compound]
#%%
df_clean3.sample(10)
#%%
df_clean3.info()
#%%
df_clean4 = df_clean3.copy()
df_clean4.text1_clean.replace('',np.nan,inplace=True)
df_clean4.text2_stopwords_stemmed.replace('',np.nan,inplace=True)
df_clean4.dropna(subset=['text2_stopwords_stemmed'],inplace=True)
df_clean4.reset_index(drop=True,inplace=True)
#%%
df_clean4.info()
#%%
df_clean5 = df_clean4.copy()
df_clean5['text1_clean_count'] = df_clean4['text1_clean'].str.len()
df_clean5 = df_clean5[df_clean5.text1_clean_count>=20]
df_clean5.reset_index(drop=True,inplace=True)
#%%
df_clean5.info()
#%%
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_clean5'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(df_clean5, to_write)
#%%
tweet1 = clean_tweet_texts[343]
tweet2 = clean_tweet_texts[1]
tweet3 = clean_tweet_texts[345]

print(tweet1)
print(tweet2)
print(tweet3)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
