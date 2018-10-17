# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:38:52 2018

@author: User
"""
import json
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import requests
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from elasticsearch import Elasticsearch
import numpy as np
#nltk.download('stopwords')

file = r'C:\Users\User\Desktop\Text Mining\Group assignment\Video_Games_5.json'

with open(file) as x:
    jsondata = pd.read_json(x, lines=True, chunksize=1000)
    df = pd.DataFrame()
    for d in jsondata:
        df = df.append(d)
x.close()
del x

df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')

reviewdf = df[['asin', 'overall', 'summary', 'reviewText', 'reviewTime']]
cols_list = ['Item', 'Stars', 'Review_Title', 'Review', 'Date']
reviewdf.columns = cols_list
print(reviewdf.head())
print(reviewdf['Review'].head())
del df

#Textblob Analysis
def sentiments(df, column):
    polarity = []
    subjectivity = []
    i = 0
    for row in df[column]:
        comm_blob = TextBlob(row)
        polarity.append(comm_blob.sentiment[0])
        subjectivity.append(comm_blob.sentiment[1])
        i += 1
        if i % 5000 == 0:
            print(i)
        else:
            continue
    df['Polarity'] = polarity
    df['Subjectivity'] = subjectivity

sentiments(reviewdf, 'Review')

#Thought this would be faster, but is slower by far
#def senti(row):
#    pol = TextBlob(row['Review']).sentiment[0]
#    sub = TextBlob(row['Review']).sentiment[1]
#    return pol, sub
#    
#
#reviewdf['Polarity'], reviewdf['Subjectivity'] = reviewdf.apply(senti, axis=1)
#
#t1= time.time()

#Kyle - make function out of Ryan's code/speed up certain lambda functions
testdf = reviewdf
def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def clean_token(df, col, lang):
    stop = stopwords.words(lang)
    df[f'{col}_Clean'] = df[col].str.replace('[^\w\s]','').str.lower()
    df[f'{col}_Clean'] = testdf['Review_Clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df[f'{col}_Token'] = testdf['Review_Clean'].apply(lemmatize_text)
    x = df[col].apply(len)
    y = df[f'{col}_Token'].apply(len)
    df['WordsRemoved'] = y - x

clean_token(reviewdf, 'Review', 'english')

#Connect to elasticsearch
es = Elasticsearch(['localhost'], port=9200)

#Create Index with specific shards
#request_body = {
#    "settings" : {
#        "number_of_shards": 1,
#        "number_of_replicas": 0
#    }
#}
#res = es.indices.create(index = 'test', body = request_body)

#ES does not recogize timestamps
reviewdf['Date'] = reviewdf['Date'].apply(lambda x: dt.datetime.strftime(x, '%m %d, %Y'))


#Send dataframe to ES
def dftoes(dataFrame, index='index', typ='test', server='http://localhost:9200',
           chunk_size=2000):
    headers = {'content-type': 'application/x-ndjson', 'Accept-Charset': 'UTF-8'}
    records = dataFrame.to_dict(orient='records')
    actions = ["""{ "index" : { "_index" : "%s", "_type" : "%s"} }\n""" % (index, typ) +json.dumps(records[j])
               for j in range(len(records))]
    i = 0
    while i < len(actions):
        serverAPI = server + '/_bulk'
        data = '\n'.join(actions[i:min([i+chunk_size, len(actions)])])
        data = data + '\n'
        r = requests.post(serverAPI, data=data, headers=headers)
        print(r.content)
        i = i+chunk_size


dftoes(reviewdf)

##Charts
# Stars distribustion
stars_count = reviewdf.groupby(['Stars'], as_index=False).count()
stars_count = stars_count[['Stars', 'Item']]
stars_count.rename(columns={'Item': 'Count'}, inplace=True)
plt.bar(stars_count['Stars'], stars_count['Count'])
plt.show()

# Mean rating overall vs mean rating by item
itemlist = list(reviewdf.Item.unique())
itemstars = reviewdf[['Item','Stars']]
allmean = reviewdf.Stars.mean()
itemmean = itemstars.groupby('Item', as_index=False).mean()
itemmeanarray = np.asarray(itemmean['Stars'])
plt.hist(itemmean['Stars'])
plt.show()


#rescale polarity to a 1-5 scale
def rescale(x, inlow, inhigh, outlow, outhigh):
    polscale = ((x - inlow) / (inhigh - inlow)) * (outhigh - outlow) + outlow
    return polscale


reviewdf['Scaled_Polarity'] = reviewdf['Polarity'].apply(rescale, args=(-1,1,1,5))
#Mean stars per item vs scaled polarity per item
scaledpoldf = reviewdf[['Item','Scaled_Polarity']]
scaledpoldf = scaledpoldf.groupby('Item', as_index=False).mean()
scaled_polarity = np.asarray(scaledpoldf['Scaled_Polarity'])
plt.xlim(1,5)
plt.ylim(1,5)
plt.xlabel('Average Stars')
plt.ylabel('Scaled Polarity')
plt.scatter(itemmeanarray, scaled_polarity, alpha=.25)


# Cumulative Reviews hist/line - Not sure why this takes so long
dtlist = sorted(list(reviewdf['Date']))
plt.hist(dtlist, density=True, histtype='step', cumulative=True, bins=1000)
