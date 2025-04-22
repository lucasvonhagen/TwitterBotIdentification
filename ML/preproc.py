import pandas as pd


def combine_data():
    #Load the bot and human tweets datasets
    bot_tweets = pd.read_csv("data/all_50_bot_tweets.csv")
    human_tweets = pd.read_csv("data/all_50_human_tweets.csv")

    #Adds a "Bot Label" column
    bot_tweets['Bot Label'] = 1  # 1 for bots
    human_tweets['Bot Label'] = 0  # 0 for humans

    #Combine the datasets
    combined_data = pd.concat([bot_tweets, human_tweets], ignore_index=True)
    return combined_data

#Create all new numeric features to use for training
def preprocess_tweet_data(data):
    
    data["Username_Digits"] = data["Twitter_User_Name"].apply(lambda x: sum(c.isdigit() for c in str(x)))

    data["Description_Length"] = data["Twitter_User_Description"].apply(lambda x: len(str(x)))

    data["Tweet_Length"] = data["Tweet_text"].apply(lambda x: len(str(x)))

    data["Tweet_Word_Count"] = data["Tweet_text"].apply(lambda x: len(str(x).split()))

    data["Tweet_Hour"] = pd.to_datetime(data["Tweet_created_at"]).dt.hour

    y = data["Bot Label"].values

    selected_features = data[["Username_Digits", "Description_Length", "Tweet_Length", "Tweet_Word_Count", "Tweet_Hour"]]

    return selected_features, y