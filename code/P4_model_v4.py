#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:55:51 2019

@author: lindaxju
"""

#%%
import pickle
from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
#from sklearn.metrics.pairwise import cosine_similarity

#from sklearn.cluster import KMeans

#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pandas.tseries.offsets import BDay
#%%
with open('data/df_clean5_2019-11-12-22-15-36.pickle','rb') as read_file:
    df_tweets = pickle.load(read_file)
#%%
df_tweets
#%%
list(df_tweets.columns)
#%%
df_tweets.info()
#%%
df_tweets.shape
#%%
list_tweets = list(df_tweets.text1_clean)
#list_tweets = list(df_tweets.text2_stopwords_stemmed)
list_tweets_label = [tweet[:30]+"..." for tweet in list_tweets]
#%%
###################################Vectorize###################################
#%%
def get_doc_word_vectorizer(vectorizer,ngram_range,list_tweets):
    if vectorizer == 'cv':
        vec = CountVectorizer(ngram_range=ngram_range,stop_words='english')
    elif vectorizer == 'tfidf':
        vec = TfidfVectorizer(ngram_range=ngram_range,stop_words='english')
    doc_word = vec.fit_transform(list_tweets)
    return vec, doc_word
#%%
def get_doc_word_df(doc_word,vectorizer,list_tweets):
    doc_word_df = pd.DataFrame(doc_word.toarray(),index=list_tweets,
                               columns=vectorizer.get_feature_names())
    return doc_word_df
#%%
ngram_range1 = (1,1)
ngram_range2 = (1,2)
ngram_range3 = (1,3)
ngram_range4 = (1,4)
#%%
# CountVectorizer
vectorizer_cv, doc_word_cv = get_doc_word_vectorizer(vectorizer='cv',ngram_range=ngram_range1,list_tweets=list_tweets)
#vectorizer_cv, doc_word_cv = get_doc_word_vectorizer(vectorizer='cv',ngram_range=ngram_range2,list_tweets=list_tweets)
#vectorizer_cv, doc_word_cv = get_doc_word_vectorizer(vectorizer='cv',ngram_range=ngram_range3,list_tweets=list_tweets)
#vectorizer_cv, doc_word_cv = get_doc_word_vectorizer(vectorizer='cv',ngram_range=ngram_range4,list_tweets=list_tweets)
get_doc_word_df(doc_word=doc_word_cv,vectorizer=vectorizer_cv,list_tweets=list_tweets)
#%%
# TF-IDF
vectorizer_tfidf, doc_word_tfidf = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=ngram_range1,list_tweets=list_tweets)
#vectorizer_tfidf, doc_word_tfidf = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=ngram_range2,list_tweets=list_tweets)
#vectorizer_tfidf, doc_word_tfidf = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=ngram_range3,list_tweets=list_tweets)
#vectorizer_tfidf, doc_word_tfidf = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=ngram_range4,list_tweets=list_tweets)
get_doc_word_df(doc_word=doc_word_tfidf,vectorizer=vectorizer_tfidf,list_tweets=list_tweets)
#%%
##########################Dim Reduction w/ LSA and NMF#########################
#%%
def get_list_components(num_components):
    list_components = []
    for num in range(num_components):
        list_components.append('c'+str(num+1).zfill(2))
    return list_components
#%%
def get_dim_red(svd,doc_word,num_components):
    if svd == 'lsa':
        dim_red = TruncatedSVD(num_components,random_state=42)
    elif svd == 'nmf':
        dim_red = NMF(num_components,random_state=42)
    doc_topic = dim_red.fit_transform(doc_word)
    return dim_red, doc_topic
#%%
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix+1)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
#%%
def get_doc_topic_df(doc_topic,cols,rnd=3):
    doc_topic_df = pd.DataFrame(doc_topic.round(rnd),columns=cols)
    doc_topic_df['topic_max'] = doc_topic_df.max(axis=1)
    doc_topic_df['cluster'] = doc_topic_df.idxmax(axis=1)
    doc_topic_df = pd.concat([df_tweets,doc_topic_df],axis=1)
    return doc_topic_df
#%%
num_components = 10
list_components = get_list_components(num_components)
#%%
# Acronynms: Latent Semantic Analysis (LSA) is just another name for 
#  Signular Value Decomposition (SVD) applied to Natural Language Processing (NLP)
lsa_tfidf, doc_topic_lsa_tfidf = get_dim_red(svd='lsa',doc_word=doc_word_tfidf,num_components=num_components)
lsa_tfidf.explained_variance_ratio_
#%%
display_topics(lsa_tfidf,vectorizer_tfidf.get_feature_names(),5)
#%%
Vt = get_doc_topic_df(doc_topic=doc_topic_lsa_tfidf,cols=list_components,rnd=3)
Vt
#%%
sorted(Counter(Vt.cluster).items())
#%%
# NMF
nmf_tfidf, doc_topic_nmf_tfidf = get_dim_red(svd='nmf',doc_word=doc_word_tfidf,num_components=num_components)
#%%
display_topics(nmf_tfidf,vectorizer_tfidf.get_feature_names(),5)
#%%
H = get_doc_topic_df(doc_topic=doc_topic_nmf_tfidf,cols=list_components,rnd=3)
H
#%%
sorted(Counter(H.cluster).items())
#%%
clusters = ['c06', 'c10']
H_filtered = H[H.topic_max >= 0.10]
H_filtered = H_filtered[H_filtered.cluster.isin(clusters)]
H_filtered = H_filtered[H_filtered['compound'].abs()>=0.7]
H_filtered.reset_index(drop=True,inplace=True)
sorted(Counter(H_filtered.cluster).items())
#%%
##################################SPY and VIX##################################
#%%
df_spy = pd.read_csv('data/GSPC.csv')
df_spy['Date_dt'] = pd.to_datetime(df_spy['Date'])
df_spy['ret_same_day'] = df_spy.Close/df_spy.Open-1
df_spy['ret_mix_day'] = df_spy.Close.shift(-1)/df_spy.Close-1
df_spy['ret_next_day'] = df_spy.Close.shift(-1)/df_spy.Open.shift(-1)-1
print(df_spy.head())
print(df_spy.info())
#%%
start_date = '01-20-2017'
end_date = '11-06-2019'
mask_SPY = (df_spy['Date_dt'] >= start_date) & (df_spy['Date_dt'] <= end_date)
df_spy_filtered = df_spy.loc[mask_SPY]
df_spy_filtered
#%%
df_vix = pd.read_csv('data/vixcurrent.csv',skiprows=1) 
df_vix.rename(columns={'VIX Open': 'VIX_Open', 'VIX Close': 'VIX_Close'},inplace=True)
df_vix['Date_dt'] = pd.to_datetime(df_vix['Date'])
df_vix['ret_same_day'] = df_vix.VIX_Close/df_vix.VIX_Open-1
df_vix['ret_mix_day'] = df_vix.VIX_Close.shift(-1)/df_vix.VIX_Close-1
df_vix['ret_next_day'] = df_vix.VIX_Close.shift(-1)/df_vix.VIX_Open.shift(-1)-1
print(df_vix.head())
print(df_vix.info())
#%%
mask_VIX = (df_vix['Date_dt'] >= start_date) & (df_vix['Date_dt'] <= end_date)
df_vix_filtered = df_vix.loc[mask_VIX]
df_vix_filtered
#%%
#################################Tweet Returns#################################
#%%
def get_business_days(df,col,df_index_dates):
    list_business_days = []
    for date in df[col]:
        if date in set(df_index_dates):
            list_business_days.append(date)
        else:
            while date not in set(df_index_dates):
                date += timedelta(days=1)
            list_business_days.append(date)
    return list_business_days
#%%
def get_time_of_day(df_tweet,col):
    list_time_of_day = []
    times = df_tweet[col]
    for time in times:
        if time.hour < 9 or time.hour == 9 and time.minute <= 30:
            time_of_day = 'before_open'
        elif time.hour <= 16:
            time_of_day = 'during_mkt_hrs'
        else:
            time_of_day = 'after_close'
        list_time_of_day.append(time_of_day)
    return list_time_of_day
#%%
def get_returns(df):
    list_ret_SPY_same_day = list(df.ret_SPY_same_day)
    list_ret_SPY_mix_day = list(df.ret_SPY_mix_day)
    list_ret_SPY_next_day = list(df.ret_SPY_next_day)
    list_rets_SPY = []
    list_ret_VIX_same_day = list(df.ret_VIX_same_day)
    list_ret_VIX_mix_day = list(df.ret_VIX_mix_day)
    list_ret_VIX_next_day = list(df.ret_VIX_next_day)
    list_rets_VIX = []
    list_times_of_day = list(df.time_of_day)
    for i in range(len(list_times_of_day)):
        if list_times_of_day[i] == 'before_open':
            list_rets_SPY.append(list_ret_SPY_same_day[i])
            list_rets_VIX.append(list_ret_VIX_same_day[i])
        elif list_times_of_day[i] == 'during_mkt_hrs':
            list_rets_SPY.append(list_ret_SPY_mix_day[i])
            list_rets_VIX.append(list_ret_VIX_mix_day[i])
        elif list_times_of_day[i] == 'after_close':
            list_rets_SPY.append(list_ret_SPY_next_day[i])
            list_rets_VIX.append(list_ret_VIX_next_day[i])
    return list_rets_SPY, list_rets_VIX
#%%
df_final1 = H_filtered.copy()

list(df_final1.columns)
#drop_cols = ['text1_clean_count','c01','c02','c03','c04','c05','c06','c07','c08',
#             'c09','c10']
drop_cols = ['neg','neu','pos','text1_clean_count','c01','c02','c03','c04','c05',
             'c06','c07','c08','c09','c10','topic_max']
df_final1.drop(drop_cols,axis=1,inplace=True)

df_final1.sort_values(by=['created_at_dt'],inplace=True)
df_final1.reset_index(drop=True,inplace=True)
print(df_final1.head())
print(df_final1.info())
#%%
df_final2 = df_final1.copy()
df_final2['created_at_dt_date'] = pd.to_datetime(df_final2['created_at_dt'].dt.date)
df_final2['day_of_week'] = df_final2['created_at_dt_date'].dt.dayofweek
df_spy_dates = df_spy.Date_dt
df_final2['business_date'] = get_business_days(df_final2,'created_at_dt_date',df_spy_dates)
df_final2['time_of_day'] = get_time_of_day(df_final2,'created_at_dt')
print(df_final2.head())
print(df_final2.info())
#%%
# SPY
df_final3 = pd.merge(df_final2,df_spy[['Date_dt','ret_same_day','ret_mix_day','ret_next_day']],
                     left_on='business_date',right_on='Date_dt',how='left')
rename_cols = {'ret_same_day':'ret_SPY_same_day','ret_mix_day':'ret_SPY_mix_day','ret_next_day':'ret_SPY_next_day'}
df_final3.rename(columns=rename_cols,inplace=True)
df_final3.drop(['Date_dt'],axis=1,inplace=True)

# VIX
df_final3 = pd.merge(df_final3,df_vix[['Date_dt','ret_same_day','ret_mix_day','ret_next_day']],
                     left_on='business_date',right_on='Date_dt',how='left')
rename_cols = {'ret_same_day':'ret_VIX_same_day','ret_mix_day':'ret_VIX_mix_day','ret_next_day':'ret_VIX_next_day'}
df_final3.rename(columns=rename_cols,inplace=True)
df_final3.drop(['Date_dt'],axis=1,inplace=True)

print(df_final3.head())
print(df_final3.info())
#%%
df_final4 = df_final3.copy()
drop_cols = ['created_at_dt','day_of_week','ret_SPY_mix_day','ret_SPY_next_day',
             'ret_VIX_mix_day','ret_VIX_next_day']
df_final4.drop(drop_cols,axis=1,inplace=True)
df_final4['returns_SPY'], df_final4['returns_VIX'] = get_returns(df_final3)
print(df_final4.head())
print(df_final4.info())
#%%
#################################Visualizations################################
#%%
def get_hist(series_hist,bins):
    plt.hist(series_hist,bins=bins)
    plt.xlabel('returns')
    plt.ylabel('count')
    plt.show();
#%%
df_final4_c06 = df_final4[df_final4.cluster == 'c06']
df_final4_c10 = df_final4[df_final4.cluster == 'c10']
df_final4_pos = df_final4[df_final4.sentiment == True]
df_final4_neg = df_final4[df_final4.sentiment == False]
#%%
print(df_final4_c06.shape)
print(df_final4_c10.shape)
print(df_final4_pos.shape)
print(df_final4_neg.shape)
#%%
bins_SPY = np.arange(-0.06,0.07,0.01)
bins_VIX = np.arange(-0.3,0.32,0.02)
#%%
def get_four_hists(s1,s1_title,s2,s2_title,s3,s3_title,s4,s4_title,bins):
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,5))
    ax1.hist(s1,bins=bins)
    ax1.set_title(s1_title)
    ax2.hist(s2,bins=bins)
    ax2.set_title(s2_title)
    ax3.hist(s3,bins=bins)
    ax3.set_title(s3_title)
    ax4.hist(s4,bins=bins)
    ax4.set_title(s4_title)
#%%
def get_six_hists(s1,s1_title,s2,s2_title,s3,s3_title,s4,s4_title,s5,s5_title,s6,s6_title,bins,filename):
    fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize=(40,5))
    ax1.hist(s1,bins=bins)
    ax1.set_title(s1_title)
    ax2.hist(s2,bins=bins)
    ax2.set_title(s2_title)
    ax3.hist(s3,bins=bins)
    ax3.set_title(s3_title)
    ax4.hist(s4,bins=bins)
    ax4.set_title(s4_title)
    ax5.hist(s5,bins=bins)
    ax5.set_title(s5_title)
    ax6.hist(s6,bins=bins)
    ax6.set_title(s6_title)
    fig.savefig(filename)
#%%
# s1 = df_spy_filtered.ret_same_day
get_four_hists(s1 = df_final4.ret_SPY_same_day,
               s1_title = 'all returns',
               s2 = df_final4.returns_SPY,
               s2_title = 'returns in clusters 6 and 10',
               s3 = df_final4_c06.returns_SPY,
               s3_title = 'returns in cluster 6',
               s4 = df_final4_c10.returns_SPY,
               s4_title = 'returns in cluster 10',
               bins = bins_SPY)  
#%%
get_four_hists(s1 = df_final4.ret_SPY_same_day,
               s1_title = 'all returns',
               s2 = df_final4.returns_SPY,
               s2_title = 'returns in clusters 6 and 10',
               s3 = df_final4_pos.returns_SPY,
               s3_title = 'returns after pos sentiment',
               s4 = df_final4_neg.returns_SPY,
               s4_title = 'returns after neg sentiment',
               bins = bins_SPY)
#%%
get_six_hists(s1 = df_spy_filtered.ret_same_day,
              s1_title = 'All Returns',
              s2 = df_final4.returns_SPY,
              s2_title = 'Returns after Macroeconomic Tweets',
              s3 = df_final4_c06.returns_SPY,
              s3_title = 'Returns after Trade/Tariff Tweets',
              s4 = df_final4_c10.returns_SPY,
              s4_title = 'Returns after Economy Tweets',
              s5 = df_final4_pos.returns_SPY,
              s5_title = 'Returns after Positive Tweets',
              s6 = df_final4_neg.returns_SPY,
              s6_title = 'Returns after Negative Tweets',
              bins = bins_SPY,
              filename = 'image4.svg') 
#%%
# s1 = df_vix_filtered.ret_same_day
get_four_hists(s1 = df_final4.ret_VIX_same_day,
               s1_title = 'all returns',
               s2 = df_final4.returns_VIX,
               s2_title = 'returns in clusters 6 and 10',
               s3 = df_final4_c06.returns_VIX,
               s3_title = 'returns in cluster 6',
               s4 = df_final4_c10.returns_VIX,
               s4_title = 'returns in cluster 10',
               bins = bins_VIX)  
#%%
get_four_hists(s1 = df_final4.ret_VIX_same_day,
               s1_title = 'all returns',
               s2 = df_final4.returns_VIX,
               s2_title = 'returns in clusters 6 and 10',
               s3 = df_final4_pos.returns_VIX,
               s3_title = 'returns after pos sentiment',
               s4 = df_final4_neg.returns_VIX,
               s4_title = 'returns after neg sentiment',
               bins = bins_VIX)
#%%
get_six_hists(s1 = df_final4.ret_VIX_same_day,
              s1_title = 'All Returns',
              s2 = df_final4.returns_VIX,
              s2_title = 'Returns after Macroeconomic Tweets',
              s3 = df_final4_c06.returns_VIX,
              s3_title = 'Returns after Trade/Tariff Tweets',
              s4 = df_final4_c10.returns_VIX,
              s4_title = 'Returns after Economy Tweets',
              s5 = df_final4_pos.returns_VIX,
              s5_title = 'Returns after Positive Tweets',
              s6 = df_final4_neg.returns_VIX,
              s6_title = 'Returns after Negative Tweets',
              bins = bins_VIX,
              filename = 'image5.svg')
#%%
def plot_time_series(df_index):
    ax = df_index.set_index('Date_dt')['Open'].plot(figsize=(20,5))
    pos_sent = df_index.loc[df_index.sentiment == True,'Date_dt']
    neg_sent = df_index.loc[df_index.sentiment == False,'Date_dt']
    for i in pos_sent:
        ax.axvline(i, color='green', alpha = 0.5, linewidth=1)
    for i in neg_sent:
        ax.axvline(i, color='red', alpha = 0.5, linewidth=1)
    plt.show()
#%%
df_spy_filtered_tweets = pd.merge(df_spy_filtered,df_final4,left_on='Date_dt',right_on='business_date',how='left')
print(df_spy_filtered_tweets.head())
print(df_spy_filtered_tweets.info())
#%%
plot_time_series(df_spy_filtered_tweets)
#%%
#%%
df_vix_filtered_tweets = pd.merge(df_vix_filtered,df_final4,left_on='Date_dt',right_on='business_date',how='left')
df_vix_filtered_tweets.rename(columns={'VIX_Open': 'Open'},inplace=True)
print(df_vix_filtered_tweets.head())
print(df_vix_filtered_tweets.info())
#%%
plot_time_series(df_vix_filtered_tweets)
#%%
df_spy_filtered_tweets2 = pd.merge(df_spy_filtered,df_final4_c10,left_on='Date_dt',right_on='business_date',how='left')
print(df_spy_filtered_tweets2.head())
print(df_spy_filtered_tweets2.info())
#%%
plot_time_series(df_spy_filtered_tweets2)
#%%
df_spy_filtered_tweets3 = pd.merge(df_spy_filtered,df_final4_c06,left_on='Date_dt',right_on='business_date',how='left')
print(df_spy_filtered_tweets3.head())
print(df_spy_filtered_tweets3.info())
#%%
plot_time_series(df_spy_filtered_tweets3)
#%%
#%%
plt.figure(figsize=[20,5])
plt.plot(df_spy_filtered.set_index('Date_dt')['Open'])
plt.savefig("image1.svg")
#%%
def plot_time_series2(df_index,filename):
    plt.rcParams.update({'font.size': 15})
    fig,ax = plt.subplots(1,figsize=(20,5))
    ax.plot(df_index.set_index('Date_dt')['Open'])
    pos_sent = df_index.loc[df_index.sentiment == True,'Date_dt']
    neg_sent = df_index.loc[df_index.sentiment == False,'Date_dt']
    for i in pos_sent:
        ax.axvline(i, color='green', alpha = 0.5, linewidth=1)
    for i in neg_sent:
        ax.axvline(i, color='red', alpha = 0.5, linewidth=1)
    fig.savefig(filename)
#%%
plot_time_series2(df_spy_filtered_tweets,"image2.svg")
#%%
plot_time_series2(df_vix_filtered_tweets,"image3.svg")
#%%
#%%
#%%
#%%
#%%
#%%

#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'df_spy_filtered_tweets'+'_'+timestamp
#df_spy_filtered_tweets.to_csv(r'data/'+filename+'.csv')
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
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
