from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# Initialise your name of Flask application as app

app = Flask(__name__)


# Define a view function named 'home', which renders the html page 'home.html'
# Ensure that the view function 'home' is routed when a user access the URL '/' .

@app.route('/')
def home():
    return render_template('home.html')


# Define a view function named 'predict', which does the function of getting the text entered by user in home.html
# and predicts if it is spam or not and renders the result in result.html Ensure that the view function 'predict' is
# routed when a user access the URL '/predict' .

'''steps
1. Load and store your vectorizer from your pickle file 'vector.pkl'
2. Load and store your classifier from your pickle file 'NB_spam_model.pkl'
3. Retrive the message given in the text area of home page.
4. Use vectoriser to fit transform the given message and store it in vect
6. Convert the data into array
7. Predict the label for the message array with the classifier loaded and store it in variable predicted_label

'''


@app.route('/predict', methods=['POST'])
def predict():
    # 1.Load the vectorizer and classifier form pickle files

    vectorizer = pickle.load(open('vector.pkl', 'rb'))
    classifier = joblib.load(open('NB_spam_model.pkl', 'rb'))

    # 2.store the inputted value retrieved from form in'message variable'
    # Hint: request.method, request.form
    message = request.form['message']
    data = [message]
    print(data)
    # 3.store the fit_transformed data in 'vect' variable
    vect = vectorizer.transform(data)
    # 4.convert it into array
    vect = vect.toarray()

    # 5.Predict the label and store in this variable
    predicted_label = classifier.predict(vect)
    if predicted_label != 0:
        predict_value = 'SPAM'
    else:
        predict_value = 'NOT A SPAM'
    return render_template('result.html', prediction=predict_value)


# make your app run in 0.0.0.0 host and port 8000
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port='8000')
