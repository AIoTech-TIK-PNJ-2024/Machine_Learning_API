import os
import joblib
import nltk
import re
import json
import cv2
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pyzbar.pyzbar import decode

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load trained model and vectorizer
model = joblib.load('sentiment.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Cleansing
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Spelling Normalization
def read_normalization_dict(file_path):
    with open(file_path, 'r') as file:
        normalization_dict = json.load(file)
    return normalization_dict

normalization_dict = read_normalization_dict('combined_slang_words.json')
def normalize_text(text, normalization_dict):
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Stopword Removal
stop_words = set(stopwords.words('indonesian'))
stop_words.discard("tidak")
def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stem_tokenizer(text):
    return [stemmer.stem(word) for word in text]

# Route Root
@app.route('/')
def root():
    return '''
            <h1>The API is ready to use</h1>
            <ul>
                <li>'/predict_sentiment' = for use sentiment analysis model</li>
            </ul>
           '''

# Route Sentiment Analysis
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    review = data['review']
    
    # Preprocess the review
    review = review.lower() # Case Folding
    review = clean_text(review)
    review = normalize_text(review, normalization_dict)
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

# Route QR Code Recognition
@app.route('/qr_recognition', methods=['POST'])
def recognize_qr():
    # Capture QR code from the request
    file = request.files['image']
    file.save('uploaded_qr.png')
    
    # Read the image
    img = cv2.imread('uploaded_qr.png')
    
    # Decode the QR code
    decoded_objects = decode(img)
    
    # Extract the data
    qr_data = [obj.data.decode('utf-8') for obj in decoded_objects]
    
    # Return the data as a JSON response
    return jsonify({'qr_data': qr_data})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)