import tweepy
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_tweet_data(tweet_url):
    """Extract tweet data and convert to features matching preproc.py"""
    #Extract tweet ID from URL
    tweet_id = tweet_url.split('/')[-1]
    
    # Initialize Twitter API client
    client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))
    
    try:
        # Get tweet and author data with all needed fields
        tweet = client.get_tweet(
            tweet_id,
            tweet_fields=['text', 'created_at'],
            expansions=['author_id'],
            user_fields=['username', 'description', 'created_at']
        )
        
        user = tweet.includes['users'][0]
        
        # Extract all features used in preproc.py
        features = {
            # From Twitter_User_Name
            "Username_Digits": sum(c.isdigit() for c in user.username),
            
            # From Twitter_User_Description
            "Description_Length": len(user.description) if user.description else 0,
            
            # From Tweet_text
            "Tweet_Length": len(tweet.data.text),
            "Tweet_Word_Count": len(tweet.data.text.split()),
            
            # From Tweet_created_at
            "Tweet_Hour": datetime.strptime(tweet.data.created_at, '%Y-%m-%dT%H:%M:%S.%fZ').hour
        }
        
        return features
    
    except Exception as e:
        raise ValueError(f"Failed to fetch tweet data: {str(e)}")

def test_scraper():
    """Test function to verify feature extraction"""
    test_url = "https://x.com/WatcherGuru/status/1909029297454780682"
    features = get_tweet_data(test_url)
    print("Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_scraper()