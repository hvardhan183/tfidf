import os
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import flask
from flask import jsonify
from flask import request
import gensim
import wikipedia
import nltk
import scipy
import csv
import json
import joblib
# base_url = "http://127.0.0.1:5129"
# def vectorizer(article_list):
#   url = base_url + "/vectorize"
#   request_data = {"articles": article_list.tolist()}
#   headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
#   response = requests.post(url, json=request_data, headers=headers)
#   response_data = response.json()
#   return response_data["vectorized"]
def tfidfvect(articles):
  tfidf_headline_vectorizer=TfidfVectorizer(min_df=0)
  try:
    tfidf_headline_vectorizer=joblib.load('tfidf_headline_vectorizer.pkl')
  except:
    news_articles = pd.read_csv("data-in-en - data-in-en.csv")
    #news_articles =news_articles.groupby('cluster').agg(lambda s: ' '.join(map(str, s)))
    #pd.DataFrame(news_articles).to_csv("data-in-en-id.csv")
    #news_articles = pd.read_csv("data-in-en-id.csv")
    news_articles_temp=news_articles.iloc[:,0:3]
    #cols = list(df.columns.values)
    #start = time.time()
    #basePath = os.getcwd()
    #out_file = os.path.join(basePath,  'final-result-en2.csv')
    #outfile = open(out_file, 'w+',encoding = 'utf-8')
    #writer = csv.writer(outfile, delimiter=',', lineterminator='\n')
    #writer.writerow(cols)
    stop_words = set(stopwords.words('english'))
    for i in range(len(news_articles_temp["text"])):
      string = ""
      for word in str(news_articles_temp.iloc[i,2]).split():
        word = ("".join(e for e in word if e.isalnum()))
        word = word.lower()
        if not word in stop_words:
          string += word + " "
      print(i)       # To track number of records processed
      news_articles_temp.iloc[i,2] = string.strip()
    lemmatizer = WordNetLemmatizer()
    for i in range(len(news_articles_temp["text"])):
      string = ""
      for w in word_tokenize(news_articles_temp.iloc[i,2]):
        string += lemmatizer.lemmatize(w,pos = "v") + " "
      news_articles_temp.iloc[i,2] = string.strip()
      print(i)
    tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
    tfidf_headline_vectorizer.fit(news_articles_temp['text'])
    joblib.dump(tfidf_headline_vectorizer,'tfidf_headline_vectorizer.pkl')
  output_list=[]
  for article in articles:
    tfidf_headline_features = tfidf_headline_vectorizer.transform([article])
    tfidf_headline_features = tfidf_headline_features.toarray()
    tfidf_headline_features = tfidf_headline_features.tolist()
    output_list.append(tfidf_headline_features)
  return output_list

app = flask.Flask(__name__)
@app.route('/vectorize', methods=['POST'])
def vectorize():
  data = request.get_json()
  articles = data['articles']
  
  #new_list=output_list.toarray()
  return jsonify(
      {"vectorized":tfidfvect(articles)}
    )

@app.route('/similar', methods=['POST'])
def similar_storyline():
  if request.method == 'POST':
    data = request.get_json()
    articles = data['articles']
    news_articles=pd.DataFrame()
    try:
      news_articles=pd.read_csv('vectors.csv')
    except:
      news_articles = pd.read_csv("data-in-en - data-in-en.csv")
      news_articles =news_articles.groupby('cluster').agg(lambda s: ' '.join(map(str, s)))
      news_articles=news_articles.reset_index()
      news_articles=news_articles.iloc[:,0:3]
      vectors=tfidfvect(news_articles['text'].to_list())
      for i in range(len(vectors)):
        vectors[i]=vectors[i][0]
      news_articles['vectors']=vectors
      news_articles.to_csv('vectors.csv',index=False,encoding='utf-8')


    #print(type(vectors))
    #print(type(vectors[0]))[]
    #print(vectors[0])
    #print(vectors.shape)
    article=""
    for i in range(len(articles)):
      article+=articles[i]
    print('h1')
    article_vector=tfidfvect([article])
    for i in range(len(article_vector)):
      article_vector[i]=article_vector[i][0]
    print('h2')
    query_embeddings = article_vector
    df=pd.read_csv("vectors.csv")
    string_series=df['vectors']
    vectors=[]
    for i in range(len(string_series)):
        list = string_series[i][1:-1].split (",")
        li = []
        for j in list:
            li.append(float(j))
        vectors.append(li)
    print('h3')
    #print(news_articles.head())
    # news_articles['vectors']=vectors
    # print(news_articles.head())
    # a=0
    # b=0
    # c=0
    # d=0
    # count=0
    # total=300
    """
    first_row=['cluster_id','title','top1','title1','top2','title2','top3','title3']
    writer.writerow(first_row)
    """
    # output only clusters with id divisible by 4

    cutoff=0.99
    df=news_articles
    i=0
    cluster_id=[]
    score=[]
    for  query_embedding in  query_embeddings:
      print(i)
      print('hi')
      distances = scipy.spatial.distance.cdist([query_embedding], vectors, "cosine")[0]
      indices = np.argsort(distances.ravel())
      idx=[]
      for k in indices:
        if distances[k]< cutoff:
          idx.append(k)
      #df1 = pd.DataFrame({'cluster': df['cluster'][idx].values,'Cosine-similarity': distances[idx].ravel()})
      cluster_id.append(df['cluster'][idx].values.tolist())
      score.append(distances[idx].ravel().tolist())
      print(idx)
      print(indices)
      print(cluster_id)
      print(type(cluster_id))
    #return jsonify({'cluster-id': cluster_id,'Cosine-similarity':score})
    return jsonify({'cluster-id': cluster_id,'Cosine-similarity':score})

     # val=int(i/4)
     # cur=1
     # temp=0
     # lst=[df['cluster'][i]]
     # lst.append(df['title'][i])
     # for j in indices[1:]:
     #     num=int(j/4)
     #     if num==val  and i%4==0 and cur==1  :
     #        count+=3
     #        temp+=1
     #     if num==val  and i%4==0 and cur==2 :
     #        count+=2
     #        temp+=1
     #     if num==val  and i%4==0 and cur==3 :
     #        count+=1
     #        temp+=1
     #     lst.append(df1['cluster'].to_list()[cur])
     #     # lst.append(df1['title'].to_list()[cur])
     #     cur+=1
     # if temp==1:
     #      a+=1
     # elif temp==2:
     #      b+=1
     # elif temp==3:
     #      c+=1
     # else:
     #      d+=1
     # if i%4==0:
     #   print(indices)
     # i+=1
     # print(count)
