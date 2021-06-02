from flask import Flask
from config import HOST, PORT
from utils.data_cleaning import clean_input
from flask import request

import pickle
import joblib

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)

lgbmc = pickle.load(open("/Users/rustemmatiev/Projects/FakeNewsClassifier/models/LGBMC.pk", "rb"))
RF = pickle.load(open("/Users/rustemmatiev/Projects/FakeNewsClassifier/models/RF.pk", "rb"))
SVM = pickle.load(open("/Users/rustemmatiev/Projects/FakeNewsClassifier/models/SVM.pk", "rb"))


@app.route("/",methods=['GET'])
def hello_world():
    return 'Service is working properly!'

@app.route("/predict",methods=['GET'])
def predict():
    req = request.get_json()
    print(req)
    input_line = clean_input(req['text'], req['title'])
    
    if 'model' in req.keys():
        if req['model'] == 'RF':
            model = RF
        elif req['model'] == 'SVM':
            model = SVM
        else:
            model = lgbmc

        return str(model.predict(input_line))

    return str(lgbmc.predict(input_line)[0])


app.run(host=HOST,port=PORT)
