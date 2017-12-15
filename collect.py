"""
collect.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import json
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from TwitterAPI import TwitterAPI


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

neg_words = []
pos_words = []

f= open("users.txt","w+")
s= open("tweets.txt","w+",encoding="utf-8")
p= open("positive.txt","w+")
n= open("negative.txt","w+")

def get_twitter():
    twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    return twitter

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter):
    result=[]
    resource='search/tweets'
    params={'q':'#iphoneX -filter:retweets', 'count':1000}
    result = robust_request(twitter,resource,params)
    return result
    pass

def friends_list(twitter,screen_name):
    friends=[]
    resource='friends/ids'
    params={'screen_name':screen_name ,'count':2000}

    following=robust_request(twitter, resource, params)
    for r in following:
        friends.append(r)
    sorted_friends=sorted(friends)
    return sorted_friends

def namesfile(twitter,names):
    friends=[]
    f.write("%s:-"%names)
    friends=friends_list(twitter,names)
    f.write(str(friends))
    f.write("\n")

def tweetsfile(tweet):
    s.write(str(tweet))
    s.write("\n")

def download_affin():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn = dict()
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    return afinn
    pass

def pos_neg_words():
   afinn=download_affin()
   pos_words=set([key for key, value in afinn.items() if value>=0])
   p.write(str(pos_words))
   neg_words=set([key for key, value in afinn.items() if value<0])
   n.write(str(neg_words))
   pass

def main():
    pos_neg_words()
    twitter = get_twitter()
    print('Established Twitter connection.')
    P_tweets = get_users(twitter)
    for u in P_tweets:
        namesfile(twitter,u['user']['screen_name'])
        tweetsfile(u['text'])
    
    f.close()
    s.close()
    p.close()
    n.close()

if __name__ == '__main__':
    main()
