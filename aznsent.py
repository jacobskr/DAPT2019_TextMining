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
#nltk.download('stopwords')

file = r'C:\Users\ejvpaba\Desktop\Python\Data\Video_Games_5.json'

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
plt.hist(itemmean['Stars'])
plt.show()

# Cumulative Reviews hist/line - Not sure why this takes so long
#dtlist = sorted(list(reviewdf['Date']))
#plt.hist(dtlist, density=True, histtype='step', cumulative=True)

#Textblob Analysis
polarity = []
subjectivity = []
i = 0
for row in reviewdf['Review']:
    comm_blob = TextBlob(row)
    polarity.append(comm_blob.sentiment[0])
    subjectivity.append(comm_blob.sentiment[1])
    i += 1
    if i % 5000 == 0:
        print(i)
    else:
        continue

reviewdf['Polarity'] = polarity
reviewdf['Subjectivity'] = subjectivity


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


#Ryan's Code:

#Make Reviews all lower case
reviewdf['Review_Clean'] = reviewdf['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Remove punctuation
reviewdf['Review_Clean'] = reviewdf['Review_Clean'].str.replace('[^\w\s]','')

#Remove english stop words
stop = stopwords.words('english')
reviewdf['Review_Clean'] = reviewdf['Review_Clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Remove the 10 most common words
freq = pd.Series(' '.join(reviewdf['Review_Clean']).split()).value_counts()[:10]

#Remove the 10 rarest words
freq = pd.Series(' '.join(reviewdf['Review_Clean']).split()).value_counts()[-10:]

#Lemmatization and Tokenization
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

nltk.download('wordnet')
reviewdf['Review_Token'] = reviewdf['Review_Clean'].apply(lemmatize_text)

#Get Review Length for both Originial Reivew and Cleaned Tokens
reviewdf['Review_Length'] = reviewdf['Review'].apply(lemmatize_text).apply(len)
reviewdf['Review_Clean_Length'] = reviewdf['Review_Token'].apply(len)

#Calculate number of words removed
reviewdf['WordsRemoved'] = reviewdf["Review_Length"] - reviewdf['Review_Clean_Length']

#End of Ryan's code


#Connect to elasticsearch
es = Elasticsearch(['localhost'], port=9200)

#Create Index
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
