import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('student_data.csv')

# Feature selection
X = df[['studytime', 'failures', 'absences', 'age', 'sex']]
y_grade = df['G3']  # Final grade

# Encode categorical data
X['sex'] = X['sex'].map({'M': 0, 'F': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_grade, test_size=0.2, random_state=42)

# Train grade prediction model
grade_model = RandomForestClassifier(n_estimators=100, random_state=42)
grade_model.fit(X_train, y_train)
joblib.dump(grade_model, 'grade_model.pkl')

# Dropout risk prediction (Assume dropout if G3 < 10)
df['dropout_risk'] = df['G3'].apply(lambda x: 1 if x < 10 else 0)  # 1 = high risk, 0 = low risk
y_dropout = df['dropout_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y_dropout, test_size=0.2, random_state=42)

dropout_model = DecisionTreeClassifier()
dropout_model.fit(X_train, y_train)
joblib.dump(dropout_model, 'dropout_model.pkl')

# Study recommendation (Categorize based on studytime)
df['study_recommendation'] = df['studytime'].apply(lambda x: 'Increase' if x < 2 else 'Maintain')
y_study = LabelEncoder().fit_transform(df['study_recommendation'])
X_train, X_test, y_train, y_test = train_test_split(X, y_study, test_size=0.2, random_state=42)

study_model = RandomForestClassifier(n_estimators=100, random_state=42)
study_model.fit(X_train, y_train)
joblib.dump(study_model, 'study_model.pkl')

print("Models trained and saved successfully!")
