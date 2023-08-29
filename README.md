🔬 **Lung Cancer Detection Project**

📅 **First Commit:** 3 weeks ago
📁 **Repository Structure:**
- `app.py`: Initial commit for the application script.
- `label_encoder_areaq.pkl` and `label_encoder_smokes.pkl`: Label encoder files for categorical variables.
- `lung-cancer-detection-92-accuracy.ipynb`: Kaggle Notebook showcasing 92% accuracy lung cancer detection.
- `lung_cancer_model.pkl`: Trained Random Forest model.

📋 **Description:**
Welcome to the Lung Cancer Detection Project repository! 🩺🦠 In this project, we explore the world of machine learning and medical diagnostics. Our goal is to detect lung cancer with high accuracy using data-driven techniques.

🔍 **Project Highlights:**
- **Dataset:** We utilize the Kaggle lung cancer dataset, loaded using pandas.
- **Data Preprocessing:** Missing values are handled, and categorical variables are label-encoded for modeling.
- **Modeling:** A Random Forest Classifier is trained on the data to achieve an accuracy of 92%.
- **Saving Models:** Trained model and label encoders are saved using joblib.
- **Evaluation:** The model's predictions are evaluated, achieving a solid accuracy score.

📊 **Code Snippet:**
```python
# Loading the dataset
data = pd.read_csv('/kaggle/input/lung-cancer-dataset/lung_cancer_examples.csv')

# Handling missing values
data.dropna(inplace=True)

# Encoding the categorical variables
label_encoder_smokes = LabelEncoder()
label_encoder_areaq = LabelEncoder()
data['Smokes'] = label_encoder_smokes.fit_transform(data['Smokes'])
data['AreaQ'] = label_encoder_areaq.fit_transform(data['AreaQ'])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}') ```python

🚀 Join us on this journey to enhance medical diagnostics using machine learning. Feel free to explore our code, contribute, and provide feedback! 🤝👩‍💻👨‍💻


