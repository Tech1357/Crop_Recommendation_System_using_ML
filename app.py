from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import os
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))

# importing model with absolute paths
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(script_dir, 'standard_scaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(script_dir, 'minmax_scaler.pkl'), 'rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    try:
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Use only StandardScaler (this gives different predictions)
        final_features = sc.transform(single_pred)
        prediction = model.predict(final_features)

        crop_name = str(prediction[0]).capitalize()
        result = "{} is the best crop to be cultivated right there".format(crop_name)
            
    except Exception as e:
        result = f"Error occurred: {str(e)}"
        
    return render_template('index.html', result=result) 




# python main
if __name__ == "__main__":
    app.run(debug=True)
