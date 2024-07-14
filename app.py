import os
import tweepy
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask import Flask, jsonify
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Twitter API Configuration
auth = tweepy.OAuthHandler(
    os.getenv('TWITTER_CONSUMER_KEY'),
    os.getenv('TWITTER_CONSUMER_SECRET')
)
auth.set_access_token(
    os.getenv('TWITTER_ACCESS_TOKEN'),
    os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
)
api = tweepy.API(auth)

# Database Configuration
Base = declarative_base()

class FashionTweet(Base):
    __tablename__ = 'fashion_tweets'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    clean_text = Column(String)
    user = Column(String)
    created_at = Column(DateTime)
    hashtags = Column(String)
    sentiment = Column(Float)

engine = create_engine(os.getenv('DATABASE_URL'))
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Fashion-related keywords
fashion_keywords = ['fashion', 'style', 'trend', 'outfit', 'clothes', 'wear', 'dress', 'shoes', 'accessories',
                    'designer', 'collection', 'runway', 'model', 'brand', 'luxury', 'vintage', 'sustainable',
                    'couture', 'vogue', 'chic', 'elegant', 'glamour', 'fashionweek', 'streetwear']

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s#]', '', text)
    return text.lower()

def extract_hashtags(text):
    return [word for word in text.split() if word.startswith('#')]

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def collect_tweets():
    session = Session()
    query = ' OR '.join(fashion_keywords)
    try:
        tweets = api.search_tweets(q=query, lang='en', count=100)
        
        for tweet in tweets:
            clean_text_content = clean_text(tweet.text)
            hashtags = extract_hashtags(clean_text_content)
            sentiment = analyze_sentiment(clean_text_content)
            
            fashion_tweet = FashionTweet(
                text=tweet.text,
                clean_text=clean_text_content,
                user=tweet.user.screen_name,
                created_at=tweet.created_at,
                hashtags=','.join(hashtags),
                sentiment=sentiment
            )
            session.add(fashion_tweet)
        
        session.commit()
        print(f"Collected and stored {len(tweets)} tweets.")
    except tweepy.TweepError as e:
        print(f"Error collecting tweets: {e}")
    finally:
        session.close()

def analyze_trends():
    session = Session()
    try:
        tweets = session.query(FashionTweet).order_by(FashionTweet.created_at.desc()).limit(1000).all()
        
        word_freq = Counter()
        hashtag_freq = Counter()
        
        for tweet in tweets:
            words = word_tokenize(tweet.clean_text)
            word_freq.update([w for w in words if w in fashion_keywords])
            hashtag_freq.update(tweet.hashtags.split(','))
        
        top_words = word_freq.most_common(10)
        top_hashtags = hashtag_freq.most_common(10)
        
        sentiment_dist = pd.Series([tweet.sentiment for tweet in tweets]).apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')).value_counts(normalize=True)
        
        return {
            'top_words': top_words,
            'top_hashtags': top_hashtags,
            'sentiment_distribution': sentiment_dist.to_dict()
        }
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        return None
    finally:
        session.close()

# Flask app for API
app = Flask(__name__)

@app.route('/generate_report', methods=['GET'])
def generate_report():
    trends = analyze_trends()
    return jsonify(trends) if trends else jsonify({"error": "Failed to generate report"}), 500

def run_collection_and_analysis():
    collect_tweets()
    report = analyze_trends()
    if report:
        print("Fashion Trend Report:")
        print("Top Words:", report['top_words'])
        print("Top Hashtags:", report['top_hashtags'])
        print("Sentiment Distribution:", report['sentiment_distribution'])
    else:
        print("Failed to generate report.")

# Schedule the task to run every 2 minutes
schedule.every(2).minutes.do(run_collection_and_analysis)

if __name__ == '__main__':
    # Run the first collection and analysis immediately
    run_collection_and_analysis()
    
    # Start the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)