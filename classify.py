# """
# classify.py
# """
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request

def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f,encoding="utf-8").readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f,encoding="utf-8").readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    if(keep_internal_punct):
      return np.array([re.sub('^\W+', '',re.sub('\W+$', '',x.lower())) for x in doc.split()])
    else:
      return np.array(re.sub('\W+', ' ', doc.lower()).split())
    pass


def token_features(tokens, feats):
    tokens_count=Counter(tokens)
    for k,v in tokens_count.items():
      feats["token="+k] = v
    pass


def token_pair_features(tokens, feats, k=3):
    token_p=[]
    temp=[]
    l=0
    for tp in tokens:
      if(l<(k-1)):
        temp.append(tp)
        l=l+1
      else:
        temp.append(tp)
        for i in range(0,len(temp)):
          for j in range((i+1),len(temp)):
            token=("token_pair="+str(temp[i])+"__"+str(temp[j]))
            feats[token]=feats[token]+1
        temp.pop(0)
    pass

pos_words=set([])
neg_words=set([])
def lexicon_features(tokens, feats):
    i=0
    j=0
    pos_words_list=[]
    neg_words_list=[]
    with open('positive.txt',encoding="utf-8") as f:
        pos_lines = f.readlines()
        pos_lines[0] = pos_lines[0].replace(" ","")
        pos_lines[0] = pos_lines[0].replace("'","")
        pos_lines[0] = pos_lines[0].replace("{","")
        pos_lines[0] = pos_lines[0].replace("}","")
        line=pos_lines[0].split(",")
        for l in range(0,len(line)):
            pos_words.add(line[l])

    with open('positive.txt',encoding="utf-8") as f:
        neg_lines = f.readlines()
        neg_lines[0] = neg_lines[0].replace(" ","")
        neg_lines[0] = neg_lines[0].replace("'","")
        neg_lines[0] = neg_lines[0].replace("{","")
        neg_lines[0] = neg_lines[0].replace("}","")
        n_line=neg_lines[0].split(",")
        for l in range(0,len(n_line)):
            neg_words.add(n_line[l])
    negative_words=0
    positive_words=0
    for t in tokens:
      if t.lower() in pos_words:
        positive_words=positive_words+1
      elif t.lower() in neg_words:
        negative_words=negative_words+1
    
    feats['pos_words']=positive_words
    feats['neg_words']=negative_words
    pass


def featurize(tokens, feature_fns):
    feats=defaultdict(lambda: 0)
    for f in feature_fns:
      f(tokens,feats)
    return sorted(feats.items(),key=lambda tup:(tup[0]))
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    data=[]
    rows=[]
    column=[]
    rw=0 
    featureslist=[]
    feats=defaultdict(lambda: 0)   
    for token in tokens_list:
        feats=featurize(token,feature_fns) 
        featureslist.append(dict(feats)) 
    if(vocab==None):
        count=defaultdict(lambda: 0)
        vocab=defaultdict(lambda: 0)
        track=defaultdict(lambda: 0)
        vocabList=[]
        for dictionary in featureslist:
            for key,value in dictionary.items():
                if dictionary[key]>0:
                    count[key]=count[key]+1
                else:
                  continue
                if (key not in track) and (count[key]>=min_freq):
                    vocabList.append(key)
                    track[key]=0
        vocabList=sorted(vocabList)
        z=0
        for key in vocabList:
                vocab[key]=z
                z+=1
    for dictionary in featureslist:
        for key,value in dictionary.items():
            if key in vocab:
                column.append(vocab[key])
                rows.append(rw)
                data.append(value)
        rw+=1
    X=csr_matrix((np.array(data,dtype='int64'), (np.array(rows,dtype='int64'),np.array(column,dtype='int64'))), shape=(rw, len(vocab)))
    return X,vocab
    pass


def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    accuracies = []
    cv = KFold(len(labels),k)
    for train_ind, test_ind in cv:
      clf.fit(X[train_ind], labels[train_ind])
      predictions = clf.predict(X[test_ind])
      accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    all_feature=[]
    clf = LogisticRegression()
    for punct in punct_vals:
      tokens_list=[]
      for d in docs:
        tokens_list.append(tokenize(d,punct)) 
      for freq in min_freqs:
        for n in range(1, len(feature_fns)+1):
          for feature in combinations(feature_fns,n):
            f_list=list(feature)
            feature_dict={}
            X,vocab=vectorize(tokens_list,f_list,freq)
            accuracy=cross_validation_accuracy(clf,X,labels,5)
            feature_dict['features']=feature
            feature_dict['punct']=punct
            feature_dict['accuracy']=accuracy
            feature_dict['min_freq']=freq
            all_feature.append(feature_dict)
    return sorted(all_feature,key=lambda k:(-k['accuracy']))
    pass


def fit_best_classifier(docs, labels, best_result):
    clf= LogisticRegression()
    alltokens=[]
    for d in docs:
      alltokens.append(tokenize(d,best_result['punct']))
    X,vocab=vectorize(alltokens,best_result['features'],best_result['min_freq'])
    clf.fit(X,labels)
    return clf,vocab
    pass

def parse_test_data(best_result, vocab,tweets):
    tokens_list = [tokenize(d,best_result['punct']) for d in tweets]
    X_test,vocab=vectorize(tokens_list,best_result['features'],best_result['min_freq'],vocab)
    return X_test
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    predicted_label = clf.predict(X_test)
    len1 = len(test_docs)
    probablities_label = clf.predict_proba(X_test)
    dictionary_list = []
    i=0
    def update_dictvalues(dictionary_list,i):
        dictionary_items = {}
        if(predicted_label[i] != test_labels[i]):
            dictionary_items['probas']= probablities_label[i]
            dictionary_items['name']= test_docs[i]
            dictionary_items['predicted']= predicted_label[i]
            dictionary_items['truth']=test_labels[i]
            dictionary_list.append(dictionary_items)
        
    while i < len1:
        update_dictvalues(dictionary_list,i)
        i+=1
    
    dict_list = sorted(dictionary_list, key=lambda x: x['probas'][x['truth']])
    for value in dict_list[:n]:
        if(value['truth']==0):
            print("truth=",value['truth'],"predicted=",value['predicted'],"proba=",value['probas'][1])
        else:
            print("truth=",value['truth'],"predicted=",value['predicted'],"proba=",value['probas'][0])
        print(value['name'])
        print()
    pass

def writeToFile(fname,classification):
    output = open(fname, 'w+',encoding="utf-8")
    for k,v in classification.items():
      output.write(k+": "+str(v))
      output.write("\n")
    output.close()


def print_top_predicted(X_test, clf, tweets):
    predicted=clf.predict(X_test)
    predictedoutput=predicted
    outputtweets=tweets[:10]
    positive_doc=[]
    negative_doc=[]
    positive_instances_count=0
    negative_instances_count=0
    classification={}
    for t in zip(predicted,tweets):
        if t[0]==0:
            negative_doc.append(t[1])
            negative_instances_count+=1
        elif t[0]==1:
            positive_instances_count+=1
            positive_doc.append(t[1])
    classification['Number of instances of Positive class']=positive_instances_count
    classification['Number of instances of Negative class']=negative_instances_count 
    classification['Example of instances of Positive Class:']=positive_doc[:1]
    classification['Example of instances of Negative Class:']=negative_doc[:1]   
    writeToFile('classification.txt',classification)
    pass

def main():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    docs, labels = read_data(os.path.join('data', 'train'))
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    worst_result = results[-1]
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    fname='tweets.txt'
    with open(fname,encoding="utf-8") as f:
        tweets = f.readlines()
    uniquetweets = set()
    for t in tweets:
        uniquetweets.add(t)
    uniquetweets=list(uniquetweets)

    X_test = parse_test_data(best_result, vocab,uniquetweets)
    print_top_predicted(X_test, clf, list(uniquetweets))


if __name__ == '__main__':
    main()

