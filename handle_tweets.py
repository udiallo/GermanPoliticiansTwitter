import os
import tweepy as tw
import pandas as pd

consumer_key= '???'
consumer_secret= '???'
access_token= '???'
access_token_secret= '???'

def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method (??? not for me)

    # authorize twitter, initialize tweepy
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, tweet_mode='extended', count=200)
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        # save most recent tweets
        new_tweets = api.user_timeline(screen_name=screen_name, tweet_mode='extended', count=200)
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))
        if len(alltweets) > max_number_of_tweets:
            break

    return alltweets
    # user = api.get_user(screen_name)
    # followers_count = user.followers_count

def tweets_to_pandas(list_of_tweets):
    columns = ["user", "id", "text", "entities", "retweet_count", "favorite_count", "retweeted"]
    tweets_frame = []
    for tweet in list_of_tweets:
        # print(tweet._json['retweeted_status']['full_text'])
        try:
            if hasattr(tweet, 'full_text'):
                if tweet.full_text[0:2] == "RT":
                    # text = 'RT ' + tweet._json['retweeted_status']['full_text']
                    data = [tweet.user.screen_name, tweet.id, "RT " + tweet._json['retweeted_status']['full_text'], tweet.entities, tweet.retweet_count,
                        tweet.favorite_count, True]
                else:
                    data = [tweet.user.screen_name, tweet.id, tweet.full_text, tweet.entities, tweet.retweet_count,
                            tweet.favorite_count, False]
            tweets_frame.append(data)
        except tw.TweepError as e:
            print(e.reason)
            continue
    df = pd.DataFrame(tweets_frame, columns=columns)
    return df

########################################################################################################################
## Main ##
users_of_interest = ["jensspahn", "CDUMerkel", "c_lindner", "ob_palmer", "HGMaassen", "Si_Wagenknecht", "ABaerbock", "Karl_Lauterbach", "Markus_Soeder", "Die_Gruenen",
                     "fdp", "spdde", "dieLinke", "CDU", "AfDBerlin"]
max_number_of_tweets = 2000 # per user

list_of_tweets = []
for user in users_of_interest:
    list_of_tweets.extend(get_all_tweets(user)) # get tweets from all users of interest
df = tweets_to_pandas(list_of_tweets) # convert tweets to dataframe

df.to_pickle("german_politicans_tweets.pkl")










# tweets = api.search(q="place:%s" % place_id, count = 1000) # more  key values possible
# for tweet in tweets:
#     #if "you" in tweet.text:
#     print(tweet.text + " | " + tweet.place.name if tweet.place else "Undefined place")


# collect tweets
#tweets = tw.Cursor(api.search,
#              q=search_words,
#              lang="en",
#              since=date_since).items(100) # restrict how many tweets. list starts with most recent tweets

# Iterate and print tweets
#for tweet in tweets:
#    print(tweet.text)

# collect a list of tweets
#list_of_tweets = [tweet.text for tweet in tweets]
#print(len(list_of_tweets))


# search_words = "#merkel"
# new_search = search_words + " -filter:retweets"
# date_since = "2021-05-25"

# userID = "CDUMerkel"
# userID = "jensspahn"
#places = api.geo_search(query="Germany", granularity="country")
#place_id = places[0].id
