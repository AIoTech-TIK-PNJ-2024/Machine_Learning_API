import pandas as pd
from flask import Flask, request, jsonify
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load('sentiment.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

stop_words = set(stopwords.words('indonesian'))

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokenizer(text):
    return [stemmer.stem(word) for word in text]


@app.route('/')
def root():
    return '''
            <h1>The API is ready to use</h1>
            <ul>
                <li>'/predict_sentiment' = for use sentiment analysis model</li>
            </ul>
           '''

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    review = data['review']
    
    # Preprocess the review
    review = review.lower()
    review = clean_text(review)
    tokens = word_tokenize(review)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokenizer(tokens)
    processed_review = ' '.join(tokens)
    
    # Vectorize the review
    review_vector = vectorizer.transform([processed_review])
    
    # Predict the sentiment
    prediction = model.predict(review_vector)
    
    # Return the prediction as a JSON response
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)