from flask import render_template, request, jsonify
from app import app
from scraper import get_tweet_data
import joblib
import os
import numpy as np

#Load model
try:
    model = joblib.load('bot_detection_model.pkl') #FIX REAL MODEL
except:
    model = None

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", title="Index")

@app.route('/analyze', methods=['POST'])
def analyze_tweet():
    data = request.get_json()
    tweet_url = data.get('url', '').strip()
    
    if not tweet_url:
        return jsonify({'message': 'No URL provided'}), 400
    
    try:
        #Get features
        features = get_tweet_data(tweet_url)
        
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
        
        #Make prediction
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