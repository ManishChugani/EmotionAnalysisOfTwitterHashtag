import re
import tweepy
from tweepy import OAuthHandler
# Fill the X's with the credentials obtained by
# following the above mentioned procedure.
consumer_key = "wUcXsXuWz2QYsAAc4Y9NyI1Rv"
consumer_secret = "GxlmiHap3YrS8Ex5H7OqEDKmUviEwzVvpgGD4i3wNJRTbB7Lrj"
access_key = "4912271577-4AqpxMwQB01r2PkLmhWnYpniyCVzlca8rQ7sPbB"
access_secret = "A6hWKxZHfu2AqN6613dT8ntZ2kRYYWVadRnZAZTp3vekX"

def clean_tweet(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Function to extract tweets
def get_tweets(hashtag):

        # Authorization to consumer key and consumer secret
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

        # Access to user's access key and access secret
        auth.set_access_token(access_key, access_secret)

        # Calling api
        api = tweepy.API(auth)

        # 200 tweets to be extracted
        number_of_tweets = 200
        #tweets = api.user_timeline(screen_name=username, count=number_of_tweets)
        tweets = []
        for tweet in tweepy.Cursor(api.search, q="#" + hashtag + "",lang="en", since="1997-01-01").items(number_of_tweets):
            tweets.append(tweet)
        # Empty Array
        tmp=[]
        # create array of tweet information: username,
        tweets_for_csv = [tweet.text for tweet in tweets]
        for j in tweets_for_csv:
            tw = clean_tweet(j)
            tw = tw.lower()
            tw = tw.split(" ")
            tmp.append(tw)
        return tmp
