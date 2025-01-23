from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

W = None
b = None

# Sigmoid function
def sig(x):
    return 1 / (1 + np.exp(-x))

# Prediction function
def predict(X, W, b):
    print(X)
    return sig(np.dot(X, W) + b)

# Load model parameters
with open("models/model_params.pkl", "rb") as f:
    model_params = pickle.load(f)
    W = model_params["Weight"]
    b = model_params["Bias"]

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form inputs
        pclass = float(request.form.get('pclass'))
        age = float(request.form.get('age'))
        sbsp = float(request.form.get('sbsp'))
        parch = float(request.form.get('parch'))
        fare = float(request.form.get('fare'))
        sex = request.form.get('sex')
        port = request.form.get('port')

        # Convert categorical inputs to numerical values
        Sex_male = 1 if sex == 'Male' else 0

        if port == "C":
            Embarked_Q = 0
            Embarked_S = 0
        elif port == "Q":
            Embarked_Q = 1
            Embarked_S = 0
        else:
            Embarked_Q = 0
            Embarked_S = 1

        # Print loaded weights and scaler details for debugging
        print("Weights are=", W)
        print("Bias is:", b)
        print("Scaler is", scaler, "min is=", scaler.min_)

        # Prepare input data
        data_dict = [{
            "Pclass": pclass,
            "Age": age,
            "SibSp": sbsp,
            "Parch": parch,
            "Fare": fare,
            "Sex_male": Sex_male,
            "Embarked_Q": Embarked_Q,
            "Embarked_S": Embarked_S
        }]
        
        df = pd.DataFrame(data_dict)
        print("Original Input DataFrame:\n", df)

        # Ensure feature names match with the scaler's expectations
        expected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        df[expected_features] = scaler.transform(df[expected_features])
        
        print("Transformed DataFrame:\n", df)
        
        # Convert to numpy array for prediction
        X = df.to_numpy()
        
        # Make prediction
        pred = predict(X, W, b)
        print("The predicted value is: ", pred)

        # Interpret prediction result
        result = "Survived" if pred[0] >= 0.5 else "Did not Survive"

        return render_template('index.html', prediction=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
