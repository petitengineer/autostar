from flask import Flask, request, jsonify
import requests
import json
import os, glob
import sqlite3
import os
import nltk
from nltk.corpus import stopwords
import re
import pickle
import numpy as np
import sys
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
from grpc.beta import implementations



###############################################################################
#
#                   FUNTIONS TO INTERACT WITH ARTIFACTS 
#
###############################################################################


def tk_preprocess_text(text, flg_stemm=False, flg_lemm=False):#, lst_stopwords=None):
  lst_stopwords=set(stopwords.words("english"))
  #Tokenizing the text into words, lowercasing, and getting rid of punctuation
  lst_text = re.findall(r'\b\w+\b', text.lower())

  # remove Stopwords
  if lst_stopwords is not None:
      lst_text = [word for word in lst_text if word not in
                  lst_stopwords]

  ## Stemming (remove -ing, -ly, ...)
  if flg_stemm == True:
      ps = nltk.stem.porter.PorterStemmer()
      lst_text = [ps.stem(word) for word in lst_text]

  ## Lemmatisation (convert the word into root word)
  if flg_lemm == True:
      lem = nltk.stem.wordnet.WordNetLemmatizer()
      lst_text = [lem.lemmatize(word) for word in lst_text]

  ## back to string from list
  text = " ".join(lst_text)
  return text

def pre_process_data(review):
  # Pre-process the review
  review = tk_preprocess_text(review)
  
  print('python version', sys.version)

  # Load the tokenizer and apply it to the review
  with open(path_to_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)
  sequence = tokenizer.texts_to_sequences([review])
  bow = tokenizer.sequences_to_matrix(sequence, mode='count')

  # Return the pre-processed review
  return bow #, tfidf

def reload_model():
  # Load the model
  champion_model = load_model(model_path)

  # Return the model
  return champion_model

#def make_prediction(model, review):
#  y_pred = model.predict(review)

#  labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

#  y_pred = tf.argmax(y_pred, axis=1)

#  return labels[y_pred.numpy()[0]]


###############################################################################
#
#                   FUNCTIONS FOR DATABASE MANIPULATION
#
###############################################################################

def create_database_and_table():
  # Connect to the SQLite database
  # If the database does not exist, it will be created
  conn = sqlite3.connect(sql_db_path)

  # Create a cursor object
  c = conn.cursor()

  # Create the table if it doesn't exist
  c.execute('''
      CREATE TABLE IF NOT EXISTS reviews (
          id INTEGER PRIMARY KEY,
          review TEXT NOT NULL,
          stars INTEGER NOT NULL
      )
  ''')

  # Commit the changes and close the connection
  conn.commit()
  conn.close()

def save_review_and_stars(review, stars):
  # Connect to the SQLite database
  conn = sqlite3.connect(sql_db_path)

  # Create a cursor object
  c = conn.cursor()

  # Check if the review already exists in the table
  c.execute('''
      SELECT * FROM reviews WHERE review = ? AND stars = ?
  ''', (review, stars))

  # If the review does not exist, insert it into the table
  if c.fetchone() is None:
      c.execute('''
          INSERT INTO reviews (review, stars)
          VALUES (?, ?)
      ''', (review, stars))

  # Commit the changes and close the connection
  conn.commit()
  conn.close()

def save_identical_review_and_stars(review, stars):
  # Connect to the SQLite database
  conn = sqlite3.connect(sql_db_path)

  # Create a cursor object
  c = conn.cursor()

  # Insert the review and stars into the table
  c.execute('''
      INSERT INTO reviews (review, stars)
      VALUES (?, ?)
  ''', (review, stars))

  # Commit the changes and close the connection
  conn.commit()
  conn.close()


def print_all_reviews():
  # Connect to the SQLite database
  conn = sqlite3.connect(sql_db_path)

  # Create a cursor object
  c = conn.cursor()

  # Execute a SELECT query to fetch all records from the reviews table
  c.execute("SELECT * FROM reviews")

  # Fetch all rows from the query
  rows = c.fetchall()

  # Print all rows
  for row in rows:
      print(row)

  # Close the connection
  conn.close()


###############################################################################
#
#                   Functions for interacting with Tensorflow Serving
#
###############################################################################

# Function to start the TensorFlow Serving server
#def start_server(model_path):
    # Command to start the server
#    command = f"tensorflow_model_server --model_base_path={model_path} --rest_api_port=8501"
#    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
#    output, error = process.communicate()

# Function to make a prediction
def make_prediction(pre_processed_review):
    # Convert the numpy array to a list
    data_list = pre_processed_review.tolist()

    # Prepare the data for the POST request
    data = json.dumps({"signature_name": "serving_default", "instances": data_list})

    # Send a POST request to the server
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://tensorflow-server:8501/v1/models/my_model:predict', data=data, headers=headers)

    # Parse the response
    response_dict = json.loads(json_response.text)

    # Check if 'predictions' key is in the response
    if 'predictions' in response_dict:
        predictions = response_dict['predictions']

        # Define the labels
        labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

        # Get the index of the highest probability
        index = np.argmax(predictions, axis=1)[0]

        # Return the label with the highest probability
        return labels[index]
    else:
        # Print the entire response if 'predictions' key is not found
        print(f"Unexpected response from the server: {response_dict}")
        return None


###############################################################################
#
#                   Creating variables, databases, etc
#
###############################################################################


nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

home_dir = os.path.expanduser("~")
sql_db_path = '/app/mysql.db'
model_path = '/app/artifacts/first_model.keras'
path_to_tokenizer = '/app/artifacts/tokenizer.pickle'

# Should create a database if one does not exist.
create_database_and_table()
champion_model = reload_model()

###############################################################################
#
#                   FUNCTIONS FOR WEBHOOKS/REQUESTS
#
###############################################################################



@app.route("/", methods=['GET', 'POST'])
def feedback_form():
    if request.method == 'POST':
        review = request.form.get('message')
        stars = int(request.form.get('stars'))
        
        pre_processed_review = pre_process_data(review)
        predicted_stars = make_prediction(pre_processed_review)

        #Add to the SQL database and print the database
        save_identical_review_and_stars(review, stars)

        #Print database (troubleshooting)
        print_all_reviews()

        print(f"Received feedback: {review} with {stars} star(s), predicted {predicted_stars} star(s).")
        return f"Thank you for your feedback! Our model found that your review: \n {review} \n should be {predicted_stars} and you rated it {stars}."

    return '''
        <form method="POST">
            <label for="message">Message:</label><br>
            <textarea id="message" name="message" rows="4" cols="50"></textarea><br>
            <label for="stars">Stars:</label><br>
            <select id="stars" name="stars">
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select><br>
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0')
