from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load models
grade_model = joblib.load('grade_model.pkl')
dropout_model = joblib.load('dropout_model.pkl')
study_model = joblib.load('study_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        age = int(request.form['age'])
        sex = int(request.form['sex'])

        # Prepare data
        input_data = pd.DataFrame([[studytime, failures, absences, age, sex]], 
                                  columns=['studytime', 'failures', 'absences', 'age', 'sex'])

        # Predict
        grade_pred = grade_model.predict(input_data)[0]
        dropout_pred = dropout_model.predict(input_data)[0]
        study_pred = study_model.predict(input_data)[0]

        dropout_risk = "High Risk" if dropout_pred == 1 else "Low Risk"
        study_suggestion = "Increase Study Time" if study_pred == 1 else "Maintain Current Study Time"

        return jsonify({
            'grade_prediction': f"Predicted Final Grade: {grade_pred}",
            'dropout_risk': f"Dropout Risk: {dropout_risk}",
            'study_recommendation': f"Study Suggestion: {study_suggestion}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
