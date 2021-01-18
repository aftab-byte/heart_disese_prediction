# -*- coding: utf-8 -*-
from flask import Flask, render_template,request
import pickle
import numpy as np
from heart import scaler
app = Flask(__name__)
model = pickle.load(open('examplePickle.pkl','rb'))
#home page
@app.route('/')
def index():
    return render_template('index.html')

#diagnosis page
@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')
#about page
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
#prediction
@app.route('/predict',methods=['POST'])
def predict():
    new_data = list(request.form.values()) 
    sex =int(new_data[2])
    new_data1 = [int(i) for i in range(1,len(new_data))]
    new_data1 =np.array(new_data1)
    new_data_reshape = new_data1.reshape(1,-1)
    new_data_scaled = scaler.transform(new_data_reshape)
    prediction = model.predict(new_data_scaled)
    return render_template('result.html',sex=sex,data=new_data,result=word(prediction))
def word(prediction):
    if prediction ==1:
        return "POSITIVE"
    else:
        return "NEGATIVE"


if __name__ =="__main__":
    app.run(debug=True)
