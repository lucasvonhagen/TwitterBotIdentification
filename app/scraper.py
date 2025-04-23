import tweepy
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_tweet_data(tweet_url):
    #Extract tweet data and convert to features
    tweet_id = tweet_url.split('/')[-1]
    
    #Initialize Twitter API client
    client = tweepy.Client(bearer_token=os.getenv('X_BEARER_TOKEN'))
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            #Get tweet and author data with all needed fields
            tweet = client.get_tweet(
                tweet_id,
                tweet_fields=['text', 'created_at'],
                expansions=['author_id'],
                user_fields=['username', 'description', 'created_at']
            )
            
            user = tweet.includes['users'][0]
            
            #Extract all features used in preproc
            features = {
                #From Twitter_User_Name
                "Username_Digits": sum(c.isdigit() for c in user.username),
                
                #From Twitter_User_Description
                "Description_Length": len(user.description) if user.description else 0,
                
                #From Tweet_text
                "Tweet_Length": len(tweet.data.text),
                "Tweet_Word_Count": len(tweet.data.text.split()),
                
                #From Tweet_created_at
                "Tweet_Hour": tweet.data.created_at.hour  # Directly access hour from datetime object
            }
            
            return features
        
        except tweepy.errors.TooManyRequests as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to fetch tweet data after {max_retries} attempts: {e}")
            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except Exception as e:
            raise ValueError(f"Failed to fetch tweet data: {e}")

def test_scraper():
    #Test function to verify scraper
    test_url = "https://x.com/WatcherGuru/status/1909029297454780682"
    try:
        features = get_tweet_data(test_url)
        print("Extracted Features:")
        for key, value in features.items():
            print(f"{key}: {value}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_scraper()