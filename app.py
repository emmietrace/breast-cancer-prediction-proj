import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load Pipeline
# This creates a robust path that works on Render and Local machines
model_path = os.path.join('model', 'breast_cancer_model.pkl')

try:
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    model_pipeline = None
    print(f"Error: Model not found at {model_path}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if not model_pipeline:
        return render_template('index.html', prediction_text="System Error: AI Model missing.")

    try:
        # 1. Get Inputs & Validation
        try:
            features = {
                'radius_mean': [float(request.form['radius_mean'])],
                'texture_mean': [float(request.form['texture_mean'])],
                'perimeter_mean': [float(request.form['perimeter_mean'])],
                'concavity_mean': [float(request.form['concavity_mean'])],
                'smoothness_mean': [float(request.form['smoothness_mean'])]
            }
        except ValueError:
             return render_template('index.html', prediction_text="Input Error: Please enter valid numeric values.")

        # 2. Create DataFrame (Pipeline expects this)
        input_df = pd.DataFrame(features)

        # 3. Predict
        prediction = model_pipeline.predict(input_df)
        prob = model_pipeline.predict_proba(input_df)[0][prediction[0]]

        # 4. Format Result
        if prediction[0] == 1:
            result = "MALIGNANT"
            css_class = "danger"
            message = f"High Risk Detected ({prob*100:.1f}% confidence). Consult a specialist immediately."
        else:
            result = "BENIGN"
            css_class = "safe"
            message = f"Tumor appears safe ({prob*100:.1f}% confidence)."

        return render_template('index.html', 
                             prediction_text=result,
                             detailed_msg=message,
                             result_class=css_class)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')