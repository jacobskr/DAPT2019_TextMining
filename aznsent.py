# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:38:52 2018

@author: Team VP
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
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
#nltk.download('stopwords')

file = 'Video_Games_5.json'

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
    """
    Input the dataframe and column that sentiment will be done on.
    Creates 2 columns:
    1. Polarity - [-1,1] How positive or negative the text is
    2. Subjectivity - [0,1] How opinionated the text is
    """
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


#Kyle - make functions out of Ryan's code/speed up certain lambda functions
def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


def clean_token(df, col, lang = 'english'):
    """ 
    Input the dataframe, column, and language (if not english) of the data
    that you want processed. This will create 3 columns:
    1. col_Clean - Lowercase text without punctuation or stopwords
    2. col_Token - col_Clean lemmatized
    3. WordsRemoved - count of words removed from original text
    """
    stop = stopwords.words(lang)
    df[f'{col}_Clean'] = df[col].str.replace('[^\w\s]','').str.lower()
    df[f'{col}_Clean'] = df[f'{col}_Clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df[f'{col}_Token'] = df[f'{col}_Clean'].apply(lemmatize_text)
    x = df[col].apply(len)
    y = df[f'{col}_Token'].apply(len)
    df['WordsRemoved'] = y - x


clean_token(reviewdf, 'Review')

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
plt.xlabel('Stars')
plt.ylabel('Count')
plt.title('Distribution of Stars Across All Games')
plt.bar(stars_count['Stars'], stars_count['Count'])
plt.show()

# Mean rating overall vs mean rating by item
itemlist = list(reviewdf.Item.unique())
itemstars = reviewdf[['Item', 'Stars']]
allmean = reviewdf.Stars.mean()
itemmean = itemstars.groupby('Item', as_index=False).mean()
itemmeanarray = np.asarray(itemmean['Stars'])
plt.xlabel('Stars - .25 Bin Width')
plt.ylabel('Count')
plt.title('Distribution of Average Star Rating per Game')
plt.hist(itemmean['Stars'], bins=16)
plt.show()

#CDF of comments over time
plt.xlabel('Year')
plt.ylabel('Comments (% of Total)')
plt.title('Amount of Comments - CDF')
reviewdf['Date'].hist(density=True, cumulative=True, bins=1000, grid=False)

#rescale polarity to a 1-5 scale
def rescale(x, inlow, inhigh, outlow, outhigh):
    polscale = ((x - inlow) / (inhigh - inlow)) * (outhigh - outlow) + outlow
    return polscale


reviewdf['Scaled_Polarity'] = reviewdf['Polarity'].apply(rescale, args=(-1, 1, 1, 5))

#Mean stars per item vs scaled polarity per item
scaledpoldf = reviewdf[['Item', 'Scaled_Polarity']]
scaledpoldf = scaledpoldf.groupby('Item', as_index=False).mean()
scaled_polarity = np.asarray(scaledpoldf['Scaled_Polarity'])
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.xlabel('Average Stars')
plt.ylabel('Scaled Polarity')
plt.scatter(itemmeanarray, scaled_polarity, alpha=.5, s=.5)
plt.annotate("x=y",
              xy=(1, 1), xycoords='data',
              xytext=(5, 5), textcoords='data',
              arrowprops=dict(arrowstyle="-",
                              edgecolor = "red",
                              linewidth=1,
                              alpha=0.65,
                              connectionstyle="arc3,rad=0."), 
              )

#Top 10th and bottom 10th percentile reviews
top10th = reviewdf[reviewdf.Polarity > reviewdf.Polarity.quantile(.90)]
bottom10th = reviewdf[reviewdf.Polarity < reviewdf.Polarity.quantile(.10)]

#Word cloud
image = r'C:\Users\ejvpaba\Desktop\Python\controller.jpg'
mask = np.array(Image.open(image))


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        mask=mask,
        background_color='black',
        stopwords=STOPWORDS,
        max_words=150,
        max_font_size=40,
        scale=3,
        random_state=1,
        colormap='gist_rainbow'
    ).generate_from_text(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20, color='white')
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(reviewdf['Review_Token'])
show_wordcloud(top10th['Review_Token'], title='Reviews - Top 10% Polarity')
show_wordcloud(bottom10th['Review_Token'], title='Reviews - Bottom 10% Polarity')
