# pip install Flask==2.0.3 Jinja2=3.0.1

from flask import Flask, render_template
from flask_restful import reqparse, Api
import flask
import pandas as pd
import os
import joblib
import ast
from model import survival_predictions
import numpy as np
import json
# how to create home route using Flask
# how to connect home route to a template view (html page)
# how to pass data from python backend to html template (Jinja engine)
# use python code in html page
# upload files to a form, send it back using url_for
# retrive files on the method that handles POST request
# get the prediction, update the context, send it back to the template

curr_path = os.path.dirname(os.path.realpath(__file__))
FEATS = joblib.load(curr_path + "/dataset/feats.pkl")
DTYPES = joblib.load(curr_path + "/dataset/dtypes.pkl")

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('list', type=list)


context_dict = {
    'feats': FEATS,
    'dtypes': DTYPES,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list,
}

@app.route('/')
def index():

    return render_template('index.html', **context_dict)

@app.route('/predict', methods=["POST"])
def predict():

    csv_file = flask.request.files['csv_file']
    test_data = pd.read_csv(csv_file)

    y_pred = survival_predictions(test_data)

    context_dict['pred']= y_pred
    context_dict['length'] = len(y_pred.tolist())

    return render_template('index.html', **context_dict)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = flask.request.form.get('single_input')
    # "[1,2,3,4]" => [1, 2, 3, 4]
    # "0" => 0
    i = ast.literal_eval(data)

    y_pred = survival_predictions(np.array(i).reshape(1,-1))
    # [0]
    
    # json: javascript object notation
    # { "" : "" }
    
    return {"message": "Success", "pred": json.dumps(int(y_pred[0]))}

if __name__ == "__main__":
    app.run()