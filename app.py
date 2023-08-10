from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained model
import joblib
model = joblib.load('lung_cancer_model.pkl')

# Load the label encoder
le_smokes = joblib.load('label_encoder_smokes.pkl')
le_areaq = joblib.load('label_encoder_areaq.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    smokes = request.form['smokes']
    areaq = request.form['areaq']
    alkhol = float(request.form['alkhol'])

    smokes_encoded = le_smokes.transform([smokes])[0]
    areaq_encoded = le_areaq.transform([areaq])[0]

    user_data = np.array([[age, smokes_encoded, areaq_encoded, alkhol]])

    prediction = model.predict(user_data)

    if prediction[0] == 1:
        result = 'High Risk of Lung Cancer'
    else:
        result = 'Low Risk of Lung Cancer'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
