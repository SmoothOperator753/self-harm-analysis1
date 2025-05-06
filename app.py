
import streamlit as st
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import snscrape

# Load tokenizer and model
with open('tokenizer_2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model("sentiment_model.keras")



import tweepy

client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAEh50QEAAAAAdGulEPuM9A1riPlL8LotxV3VfWk%3DjJ9DUUqYflaMGiQAHNnqzTfmXphRh1o4KpxWZ8oMMtF1n0dq4D')
# Replace 'USER_ID' with the actual user ID




# Function to preprocess input text
def preprocess_text(text):
    text = text.lower()
    #text = word_tokenize(text)
    return " ".join(text)

# Prediction function
def predict_text(text):
    #text = preprocess_text(text)
    token_list = tokenizer.texts_to_sequences([text])
    token_padded = pad_sequences(token_list, maxlen=200, padding='pre', truncating='pre')
    prediction = model.predict(token_padded)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

sentiment_mapping = {
      0: 'Anxiety',
      1: 'Bipolar',
      2: 'Depression',
      3: 'Normal',
      4: 'Personality Disorder',
      5: 'Stress',
      6: 'Suicidal'
  }


# Streamlit UI
import streamlit as st
from collections import Counter

# Function to fetch last 5 tweets of a user
def fetch_last_5_tweets(user_id):
    tweets = client.get_users_tweets(id=user_id, max_results=5, tweet_fields=['created_at', 'text'])
    return [tweet.text for tweet in tweets.data] if tweets.data else []

# Function to predict sentiment for multiple tweets
def predict_majority_sentiment(tweets):
    sentiments = [predict_text(tweet) for tweet in tweets]  # Predict sentiment for each tweet
    sentiment_counts = Counter(sentiments)  # Count occurrences of each sentiment
    majority_sentiment = sentiment_counts.most_common(1)[0][0]  # Get the most common sentiment
    return majority_sentiment

# Streamlit UI
st.title("Twitter Sentiment Analysis")

user_input = st.text_input("Enter Twitter User ID:")

if st.button("Fetch & Predict"):
    if user_input:
        last_5_tweets = fetch_last_5_tweets(user_input)  # Fetch last 5 tweets

        if last_5_tweets:
            st.subheader("Last 5 Tweets:")
            for i, tweet in enumerate(last_5_tweets, 1):
                st.write(f"**Tweet {i}:** {tweet}")  # Display each tweet

            # Predict sentiment for each tweet
            majority_sentiment = predict_majority_sentiment(last_5_tweets)
            st.success(f"Majority Sentiment: {sentiment_mapping[majority_sentiment]}")  # Print final sentiment
        else:
            st.warning("No tweets found for this user.")
    else:
        st.warning("Please enter a valid Twitter User ID.")


