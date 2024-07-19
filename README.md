
# Diabetes Prediction Model

This project involves creating a machine learning model to predict diabetes based on patient data using the PIMA Diabetes Dataset. The model is built using Python and various machine learning libraries, including NumPy, Pandas, and scikit-learn.

## Table of Contents
- [Data Collection and Analysis](#data-collection-and-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [Requirements](#requirements)

## Data Collection and Analysis
The PIMA Diabetes Dataset is loaded into a pandas DataFrame for analysis. The dataset contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The dataset is examined to understand its structure and statistical properties.

## Data Preprocessing
The data is preprocessed by separating the features (X) and the target variable (y). The features are then standardized to have a mean of 0 and a variance of 1 using `StandardScaler` from scikit-learn.

## Model Training
The preprocessed data is split into training and testing sets using `train_test_split`. An SVM (Support Vector Machine) classifier with a linear kernel is trained on the training data.

## Model Evaluation
The trained model is evaluated using accuracy scores on both the training and testing datasets. The accuracy scores are printed to assess the model's performance.

## Saving and Loading the Model
The trained model is saved to a file using the `pickle` module. The saved model can be loaded later for making predictions.

## Making Predictions
A predictive system is created to classify new input data. The input data is transformed to match the standardization applied during training, and the model predicts whether the person is diabetic or not.

## Usage
1. Clone the repository.
2. Install the required libraries using the command: `pip install -r requirements.txt`.
3. Run the script to train the model and evaluate its performance.
4. Use the saved model to make predictions on new data.

## Requirements
- Python 3.6+
- NumPy
- Pandas
- scikit-learn

## Example Code

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Separate features and target variable
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, stratify=y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Evaluate the model
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

x_test_prediction = classifier.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy score of the training data:', training_data_accuracy)
print('Accuracy score of the testing data:', testing_data_accuracy)

# Save the model
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Load the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Make a prediction
input_data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)
input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = loaded_model.predict(std_data)

if prediction[0] == 0:
    print("This person is not diabetic")
else:
    print("This person is diabetic")
```

This README provides an overview of the project and guides users on how to use the provided code to train a model and make predictions. Make sure to adjust any file paths and dependencies as needed.
