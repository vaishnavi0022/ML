from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('heart.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('heart_before.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['chestPainType']
    data4 = request.form['restingBP']
    data5 = request.form['cholesterol']
    data6 = request.form['fastingBS']
    data7 = request.form['restingECG']
    data8 = request.form['maxHR']
    data9 = request.form['exerciseAngina']
    data10 = request.form['oldpeak']
    data11 = request.form['stSlope']
    
    arr = np.array([[data1, data2, data3, data4 , data5, data6, data7, data8, data9 , data10 , data11]])
    pred = model.predict(arr)
    return render_template('heart_after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















