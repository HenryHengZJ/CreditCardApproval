import json
import sklearn
import os
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from sklearn.ensemble import RandomForestClassifier

json_data = {"Gender":0.0,"Reality":1.0,"ChldNo_1":0.0,"ChldNo_2More":0.0,"wkphone":0.0,"gp_Age_high":0.0,"gp_Age_highest":0.0,"gp_Age_low":0.0,"gp_Age_lowest":0.0,"gp_worktm_high":0.0,"gp_worktm_highest":0.0,"gp_worktm_low":0.0,"gp_worktm_medium":0.0,"occyp_hightecwk":0.0,"occyp_officewk":1.0,"famsizegp_1":0.0,"famsizegp_3more":0.0,"houtp_Co-op apartment":0.0,"houtp_Municipal apartment":0.0,"houtp_Office apartment":0.0,"houtp_Rented apartment":0.0,"houtp_With parents":0.0,"edutp_Higher education":0.0,"edutp_Incomplete higher":0.0,"edutp_Lower secondary":0.0,"famtp_Civil marriage":0.0,"famtp_Separated":0.0,"famtp_Single \/ not married":0.0,"famtp_Widow":0.0}
json_data2= {"Gender":1.0,"Reality":0.5061603761,"ChldNo_1":1.0,"ChldNo_2More":0.0,"wkphone":0.0,"gp_Age_high":0.5061603761,"gp_Age_highest":0.0,"gp_Age_low":0.0,"gp_Age_lowest":0.0,"gp_worktm_high":0.0,"gp_worktm_highest":0.0,"gp_worktm_low":0.5061603761,"gp_worktm_medium":0.0,"occyp_hightecwk":1.0,"occyp_officewk":0.0,"famsizegp_1":0.0,"famsizegp_3more":1.0,"houtp_Co-op apartment":0.0,"houtp_Municipal apartment":0.0,"houtp_Office apartment":0.0,"houtp_Rented apartment":0.0,"houtp_With parents":0.0,"edutp_Higher education":1.0,"edutp_Incomplete higher":0.0,"edutp_Lower secondary":0.0,"famtp_Civil marriage":0.0,"famtp_Separated":0.0,"famtp_Single \/ not married":0.0,"famtp_Widow":0.0}

app = Flask(__name__)
api = Api(app)

class Classifier(Resource):

    def post(self):

        inputDF = pd.json_normalize(request.json)

        with open('d:/ML/creditcard/credit_card_model.p', 'rb') as f:
            model = pickle.load(f)

        # Predict class
        try:
            prediction = model.predict_proba(inputDF)
            prediction = prediction[0]
            predicted_probability = max(prediction)
            predicted_class = [i for i, v in enumerate(prediction) if v == predicted_probability]
            if predicted_class == [0]:
                predicted_class = "Good"
            else:
                predicted_class = "Risky"
            
            return {"message": "Successful prediction", "class": predicted_class, "probability": predicted_probability}, 200
        except:
            return {"message": "Error predicting"}, 409

api.add_resource(Classifier, '/classify')  # add endpoints

if __name__ == '__main__':
    app.run()  # run our Flask app