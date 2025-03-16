import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('encoder.pk1', 'rb'))

@app.route('/') # route to display the home page
def home():
    return render_template('index.html') # rendering the home page


@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Read input values from the form
    input_feature = [float(x) for x in request.form.values()]

    # Convert input to a NumPy array
    features_values = [np.array(input_feature) ] # Ensure it's a 2D array

    # Define column names properly
    names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day', 'hours', 'minutes', 'seconds']]

    # Create a DataFrame
    data = pandas.DataFrame(features_values, columns=names)



    # Make prediction
    prediction = model.predict(data)

    print(prediction)
    text = "Estimated Traffic Volume is: "

    return render_template("output.html", prediction_text=text + str(prediction))


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8000, debug=True) # running the app
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)