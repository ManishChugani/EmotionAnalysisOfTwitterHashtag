# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('text_emotion2.csv')

# Cleaning the texts
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import GetTweets as GT
corpus = []
hashtag = input("Enter the hashtag of which you want to search Tweets \n")
cleaned_hashtag = hashtag.split("#")
if("#" in cleaned_hashtag):
    cleaned_hashtag.remove("#")
for i in range(len(cleaned_hashtag)):
    cleaned_hashtag_str = "".join(cleaned_hashtag[i])
tweets = GT.get_tweets(cleaned_hashtag_str)
for i in range(0, 10000):
    review = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/ \/ \S+)', ' ', dataset['content'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
length_tweets = len(tweets)
if(length_tweets > 100):
    length_tweets = 100
for i in range(0, length_tweets):
    # ps = PorterStemmer()
    # review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]'''
    retweet = 'rt'
    if retweet in tweets[i]:
        tweets[i].remove(retweet)
    tweet = [word for word in tweets[i] if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:10000, 1].values
X_train = X[:10000, :]
X_test = X[10000:, :]
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y)
Tweets_Display = []
Emotions_Display = []
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from prettytable import PrettyTable
t = PrettyTable(["Tweet", "Emotion"])
t.align["Tweet"] = "l"
for i in range(10):
    t.add_row([(" ".join(tweets[i])), y_pred[i]])
print(t)

#PieChart Calculations
happy, sad, angry, surprise, neutral = 0, 0, 0, 0, 0
for emotion in y_pred:
    if emotion == "happy":
        happy += 1
    elif emotion == "sad":
        sad += 1
    elif emotion == "angry":
        angry += 1
    elif emotion == "surprise":
        surprise += 1
    elif emotion == "neutral":
        neutral += 1

#matplotlib Plotting the PieChart
import matplotlib.pyplot as plt
labels = 'Happy', 'Sad', 'Angry', 'Surprise', 'Neutral'
sizes = [happy, sad, angry, surprise, neutral]
colors = ['gold', 'lightskyblue', 'red', 'orange', 'grey']
# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0)

plt.axis('equal')
plt.show()
