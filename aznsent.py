# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:38:52 2018

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = r'C:\Users\Kyle Jacobs\Desktop\Python\Data\Video_Games_5.json'
df = pd.read_json(file, lines = True)
cols = list(df.columns)
df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%m %d, %Y')

reviewdf = df[['asin', 'overall', 'summary', 'reviewText', 'reviewTime']]
cols_list = ['Item', 'Stars', 'Review_Title', 'Review', 'Date']
reviewdf.columns = cols_list
print(reviewdf.head())
print(reviewdf['Review'].head()) 

# Stars distribustion
stars_count = reviewdf.groupby(['Stars']).count()
stars_count = stars_count.iloc[:,0:1]
stars_count.columns = ['Count']
plt.bar(stars_count.index, stars_count['Count'])
plt.show()

reviewcount = reviewdf.groupby(['Date']).count()
plt.bar(reviewcount.index, reviewcount['Item'])
min(reviewdf['Date'])
max(reviewdf['Date'])
max(reviewdf['Date']) - min(reviewdf['Date'])

dtlist = sorted(list(reviewdf['Date']))

plt.hist(dtlist, density = True, histtype = 'step', cumulative = True)

 