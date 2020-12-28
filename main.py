from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,StandardScaler
from keras.models import load_model
import tensorflow as tf
from ml_final import prepare_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def post_submit():
    json_ = request.form.to_dict()
    print(request.form)
    df = pd.DataFrame([json_])
    X_std = StandardScaler().fit_transform(df)
    pca,x_pca = prepare_data()
    query_df = pca.transform(X_std)
    # with graph.as_default():        
    prediction = clf.predict(query_df)
    print(prediction)
    if prediction == 1:
        return render_template('result.html', prediction_text='Satisfaction')
    else:
        return render_template('result.html', prediction_text='neutral or dissatisfaction')

@app.route('/result', methods=['POST'])
def back():
    return render_template('index.html')
    
if __name__ == "__main__":
    clf = pickle.load(open('model.pkl','rb'))
    # model = load_model('model.h5')
    # model._make_predict_function()
    # graph = tf.get_default_graph()
    app.run()