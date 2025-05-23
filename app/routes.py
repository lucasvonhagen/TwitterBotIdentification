from flask import render_template, request, jsonify
from app import app
from . import scraper
import joblib
import os
import numpy as np

#Load model
try:
    model = joblib.load('bot_detection_model.pkl')
except:
    model = None

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title="Index")

@app.route('/analyze', methods=['POST'])
def analyze_tweet():
    data = request.get_json()
    print(f"DATA | {data}")

    tweet_url = data.get('url', '').strip()
    print(f"URL | {tweet_url}")
    
    if not tweet_url:
        return jsonify({'message': 'No URL provided'}), 400
    
    try:
        #Get features
        features = scraper.get_tweet_data(tweet_url)
        
        #Prepare features
        feature_order = [
            'Username_Digits',
            'Description_Length',
            'Tweet_Length',
            'Tweet_Word_Count',
            'Tweet_Hour'
        ]
        
        #Convert features to array in the correct order
        feature_array = np.array([[features[feature] for feature in feature_order]])
        print(f"FEATURES | " + str({feature_array}))
        
        #Make prediction, not finished
        if model:
            bot_probability = model.predict_proba(feature_array)[0][1]
            print("PROB: " + bot_probability)
            is_bot = bot_probability > 0.5
        else:
            #Testing
            bot_probability = 0.75  # Example value
            is_bot = True
        
        return jsonify({
            'bot_probability': float(bot_probability),
            'is_bot': bool(is_bot),
            'features': features
        })
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500