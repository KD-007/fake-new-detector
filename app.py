from flask import Flask , request , jsonify , render_template
import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import joblib



app = Flask(__name__)

model = None
vectorizer = None

def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

@app.route('/' , methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json()
    print(data)
    preprocessed_text = preprocess_text(data['data'])  # Extract preprocessed text from the list
    preprocessed_text = vectorizer.transform(np.array([preprocessed_text])).toarray() 
    prediction = model.predict(preprocessed_text)
    
    prediction = (prediction > 0.5).astype(int)
    
    result = "Fake" if prediction[0] == 1 else "Not Fake"
    
    response = {
        'prediction': result
    }
    
    return jsonify(response)
    
    
if __name__ == '__main__':
    model = tf.keras.models.load_model('keras_model.h5')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("app is running")
    
    app.run(debug=True)