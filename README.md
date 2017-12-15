# Sentiment-Analysis

In this implementation, I found customer sentiments about iphoneX that just released worldwide. This is currently trending product worldwide and people have both positive and negative sentiments about it.

I am collecting tweets which has #iphoneX using search API of twitter and classifying them as Positive and negative peoples sentiment based on training data which will train model.

In collect.py, I am finding out users who has tweeted about iphoneX and using there id, I am also Collecting whom they follow.

In cluster.py, using data from collect.py I am creating graph which will be one big component. This graph I am dividing in different communities using girvan newman. This information I am using to find out average user per community.

In classify.py I am finding out positive and negative sentiments of people from tweets collected. To find out positive and negative sentiments of people, first I am training my model using train data which is classified using AFINN. Then I am collecting live twitter data which will be my test data, and then I am classifying this data as positive and negative sentiments.

In summarize.py file I am showing final analysis. By analyzing data in file we found more people having negative sentiments about iphoneX and people are unhappy about the new apple phone while there are some people who loved the phone and having positive sentiment.
