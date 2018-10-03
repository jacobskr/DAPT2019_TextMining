# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:17:07 2018

@author: ejvpaba
"""
import datetime as dt
import csv
import praw
from praw.models import MoreComments
import pandas as pd
from textblob import TextBlob
from elasticsearch import helpers, Elasticsearch

def hsrs(subreddit, es_index = '', es_doctype = '', amount = 10,
         cl_id = '', cl_secret = '', usr_agent = '', usrnm = '', pword = ''):
    """Gets "hot" reddit posts under given Subreddit, saves data about the topic,
    comments, and replies with a sentiment analysis for each. Creates a CSV with
    the data and uploads to ElasticSearch if desired.
    
    Required arguments: subreddit = which subreddit to scrape
                        es_index = ElasticSearch index to load to
                        es_doctype = ElasticSearch doctype to load to
    Default arguments: amount = default 10. Number of topics to return
                        
    Requires a config.py file with client_id, client_secret, user_agent, username,
    and password. If no config file, pass these as arguements for cl_id, cl_secret,
    usr_agent, usrnm, and pword.
    
    Learn more about praw at https://praw.readthedocs.io/en/latest/"""
 
    # Try importing config, if no config, use inputs to connect to praw.
    if cl_id == '' and cl_secret == '' and usr_agent == '' and usrnm == '' and pword == '':
        import config
        REDDIT = praw.Reddit(client_id = config.client_id,
                             client_secret = config.client_secret,
                             user_agent = config.user_agent,
                             username = config.username,
                             password = config.password)
    else:
        REDDIT = praw.Reddit(client_id= cl_id,
                             client_secret = cl_secret,
                             user_agent = usr_agent,
                             username = usrnm,
                             password = pword)
            
    
    SUBREDDIT = REDDIT.subreddit(subreddit)
    
    HOT_SUBREDDIT = SUBREDDIT.hot(limit = amount)

    # Initialize and define keys in dictionary.
    HOTREDDIT_DICT = {"topic_id": [], "topic": [], "score": [], "topic_body": [],
                      "url": [], "num_comms": [], "topic_created": [],
                      "comm_id":[], "comment": [], "comm_created": [],
                      "comm_polarity": [], "comm_subjectivity": [],
                      "rep_id":[], "rep":[], "rep_created":[], "rep_polarity":[],
                      "rep_subjectivity":[]}
    
    # Append dictionary with all topics, comments, replies
    ITERATION = 1
    for submission in HOT_SUBREDDIT:
        ITERATION += 1
        submission = REDDIT.submission(id=submission.id)
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            for second_level_comment in top_level_comment.replies:
                if isinstance(second_level_comment, MoreComments):
                    continue
                # Use TextBlob for comments and replies to extract sentiment
                comms_blob = TextBlob(str(top_level_comment.body))
                rep_blob = TextBlob(str(second_level_comment.body))
                HOTREDDIT_DICT["topic_id"].append(submission.id)
                HOTREDDIT_DICT["topic"].append(submission.title)
                HOTREDDIT_DICT["score"].append(submission.score)
                HOTREDDIT_DICT["topic_body"].append(submission.selftext)
                HOTREDDIT_DICT["url"].append(submission.url)
                HOTREDDIT_DICT["num_comms"].append(submission.num_comments)
                HOTREDDIT_DICT["topic_created"].append(submission.created)
                HOTREDDIT_DICT["comm_id"].append(top_level_comment)
                HOTREDDIT_DICT["comment"].append(top_level_comment.body)
                HOTREDDIT_DICT["comm_created"].append(top_level_comment.created)
                HOTREDDIT_DICT["comm_polarity"].append(comms_blob.sentiment[0])
                HOTREDDIT_DICT["comm_subjectivity"].append(comms_blob.sentiment[1])
                HOTREDDIT_DICT['rep_id'].append(second_level_comment)
                HOTREDDIT_DICT['rep'].append(second_level_comment.body)
                HOTREDDIT_DICT["rep_created"].append(second_level_comment.created)
                HOTREDDIT_DICT["rep_polarity"].append(rep_blob.sentiment[0])
                HOTREDDIT_DICT["rep_subjectivity"].append(rep_blob.sentiment[1])
    
    # Dict to df for CSV file
    HOTREDDIT_DATA = pd.DataFrame(HOTREDDIT_DICT)
    
    
    # Change timestamps into readable datetime
    def _get_date_(created):
        return dt.datetime.fromtimestamp(created)
    
    
    _TIMESTAMP1 = HOTREDDIT_DATA["topic_created"].apply(_get_date_)
    HOTREDDIT_DATA = HOTREDDIT_DATA.assign(timestamp=_TIMESTAMP1)
    
    _TIMESTAMP2 = HOTREDDIT_DATA["comm_created"].apply(_get_date_)
    HOTREDDIT_DATA = HOTREDDIT_DATA.assign(timestamp=_TIMESTAMP2)
    
    _TIMESTAMP3 = HOTREDDIT_DATA["rep_created"].apply(_get_date_)
    HOTREDDIT_DATA = HOTREDDIT_DATA.assign(timestamp=_TIMESTAMP3)

    # Saves as CSV with the form r + subreddit chosen + today's date    
    HOTREDDIT_DATA.to_csv(f'r{subreddit}_'+str(dt.datetime.now().strftime('%Y-%m-%d'))+'.csv', index=False)

    # Upload to ElasticSearch if desired (requires non-default args)
    if es_index == '' and es_doctype == '':
        pass

    else:
        ES = Elasticsearch(['localhost'], port=9200)
        
        with open(f'r{subreddit}_'+str(dt.datetime.now().strftime('%Y-%m-%d'))+'.csv', encoding='utf-8') as x:
            READER = csv.DictReader(x)
            helpers.bulk(ES, READER, index=es_index, doc_type=es_doctype)