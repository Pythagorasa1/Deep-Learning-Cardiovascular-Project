"""
Combined Prototype File for Multimodal Deep Learning for Cardiovascular Prediction

This file includes functions to load and process data, build models, evaluate them, and 
an API to serve predictions. Each section is commented with explanations in simple terms.

Usage:
    To run training/evaluation: python combined_prototype.py train
    To run the API server:      python combined_prototype.py api
"""

#######################################
#             IMPORTS                 #
#######################################
import os                                  # For operating system interactions
import sys                                 # For reading command-line arguments
import requests                            # To talk to other web services
import pandas as pd                        # For working with tables (data frames)
import numpy as np                         # For math and numbers
import missingno as msno                   # To visualize missing data
from typing import Optional                # To say that something may be optional
from sklearn.impute import SimpleImputer   # To fill in missing numbers
from pyspark.sql import SparkSession       # To use Spark (for big data)
from pyspark.sql.types import StructType   # To define the structure of data
from pyspark.sql import DataFrame  # Spark DataFrame type
from tsfresh import extract_features        # To get extra details from time series data

# TensorFlow and Keras for building deep learning models
import tensorflow as tf
from tensorflow import keras
Model = keras.models.Model
Sequential = keras.models.Sequential
load_model = keras.models.load_model
layers = tf.keras.layers
Input = layers.Input
Dense = layers.Dense
Dropout = layers.Dropout
BatchNormalization = layers.BatchNormalization
Conv2D = layers.Conv2D
LSTM = layers.LSTM
Flatten = layers.Flatten
Concatenate = layers.Concatenate
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping
to_categorical = tf.keras.utils.to_categorical

import joblib                              # For saving/loading objects (like our scaler)
import matplotlib.pyplot as plt            # For plotting graphs
import shap                                # For explaining model predictions

# Scikit-learn tools for evaluation and data splitting
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pydantic for validating incoming API data
from pydantic import BaseModel, ValidationError

# Flask for the API
from flask import Flask, request, jsonify

#######################################
#       DATA PROCESSING FUNCTIONS     #
#######################################

def get_spark_session(app_name: str = "MultimodalDataPipeline") -> SparkSession:
    """
    Start a Spark session (like turning on a big helper computer).
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.extraJavaOptions",
                "-Djava.security.manager -Djava.security.policy=file:///Users/manny/Desktop/Desktop%20Folder/MultimodalDataset021525/spark.policy.txt") \
        .getOrCreate()

def load_data_spark(file_path: str, schema: Optional[StructType] = None) -> DataFrame:
    """
    Load a CSV file using Spark and cache it (store it for fast access).
    
    Args:
        file_path (str): The path to the CSV file.
        schema (Optional[StructType]): The structure of the data (if provided).
        
    Returns:
        DataFrame: A Spark DataFrame with the data.
    """
    spark = get_spark_session()
    # Read the CSV file with header, guess the schema, and cache the result.
    return spark.read.csv(r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/ehr_data.csv', header=True, schema=schema, inferSchema=True).cache()


def load_data_spark(file_path: str, schema: Optional[StructType] = None) -> DataFrame:
    """
    Load a CSV file using Spark and cache it (store it for fast access).
    
    Args:
        file_path (str): The path to the CSV file.
        schema (Optional[StructType]): The structure of the data (if provided).
        
    Returns:
        DataFrame: A Spark DataFrame with the data.
    """
    spark = get_spark_session()
    # Read the CSV file with header, guess the schema, and cache the result.
    return spark.read.csv(r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/imaging_data.csv', header=True, schema=schema, inferSchema=True).cache()
def load_data_spark(file_path: str, schema: Optional[StructType] = None) -> DataFrame:
    """
    Load a CSV file using Spark and cache it (store it for fast access).
    
    Args:
        file_path (str): The path to the CSV file.
        schema (Optional[StructType]): The structure of the data (if provided).
        
    Returns:
        DataFrame: A Spark DataFrame with the data.
    """
    spark = get_spark_session()
    # Read the CSV file with header, guess the schema, and cache the result.
    return spark.read.csv(r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/wearables_data.csv', header=True, schema=schema, inferSchema=True).cache()
def load_data_spark(file_path: str, schema: Optional[StructType] = None) -> DataFrame:
    """
    Load a CSV file using Spark and cache it (store it for fast access).
    
    Args:
        file_path (str): The path to the CSV file.
        schema (Optional[StructType]): The structure of the data (if provided).
        
    Returns:
        DataFrame: A Spark DataFrame with the data.
    """
    spark = get_spark_session()
    # Read the CSV file with header, guess the schema, and cache the result.
    return spark.read.csv(r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/genomic_data.csv', header=True, schema=schema, inferSchema=True).cache()

def impute_and_visualize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show missing data with a plot and fill in missing numbers with the average.
    
    Args:
        df (pd.DataFrame): The table of data.
        
    Returns:
        pd.DataFrame: The same table with missing numbers filled.
    """
    msno.matrix(df)  # Show a picture of missing data
    
    def impute_and_visualize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Plot missing values and impute numeric columns with the mean.
        Non-numeric columns remain unchanged.
        """
        # Plot missing data (you may choose to plot the whole df)
    msno.matrix(df)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        # Impute numeric columns only
        imputer = SimpleImputer(strategy='mean')
        df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
        # Replace numeric columns in the original df with imputed values
        df.loc[:, numeric_cols] = df_numeric
    return df

def merge_data(ehr_data: pd.DataFrame, imaging_data: pd.DataFrame, 
               wearables_data: pd.DataFrame, genomic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine different tables into one big table using the 'Patient_ID' column.
    
    Args:
        ehr_data, imaging_data, wearables_data, genomic_data (pd.DataFrame): The data tables.
        
    Returns:
        pd.DataFrame: The merged table.
    """
    # Function to rename non-key columns by adding a modality prefix
    def add_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return df.rename(columns=lambda x: x if x == 'Patient_ID' else f"{prefix}_{x}")

    # Rename columns for imaging, wearables, and genomic data.
    imaging_data = add_prefix(imaging_data, 'imaging')
    wearables_data = add_prefix(wearables_data, 'wearables')
    genomic_data = add_prefix(genomic_data, 'genomic')
    
    # Merge the dataframes on the common key "Patient_ID"
    merged = ehr_data.merge(imaging_data, on='Patient_ID')\
                     .merge(wearables_data, on='Patient_ID')\
                     .merge(genomic_data, on='Patient_ID')
    return merged

def extract_time_series_features(wearables_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract extra details from time-series data (like steps over time) using tsfresh.
    
    Args:
        wearables_data (pd.DataFrame): Wearable device data with 'Patient_ID' and 'time'.
        
    Returns:
        pd.DataFrame: A table with extra features.
    """
    # If no "time" column exists, create one (e.g., using the row index as a dummy time).
    if 'time' not in wearables_data.columns:
        wearables_data = wearables_data.copy()
        wearables_data['time'] = range(1, len(wearables_data) + 1)
        
    return extract_features(wearables_data, column_id="Patient_ID", column_sort="time")

#######################################
#       MODEL BUILDING FUNCTIONS      #
#######################################

def build_deep_learning_model(input_dim: int) -> tf.keras.Model:
    """
    Build a model that predicts both a classification (yes/no disease) and regression (time).
    
    Args:
        input_dim (int): The number of input features.
        
    Returns:
        Model: A compiled Keras model.
    """
    # Create the input layer expecting input_dim numbers.
    inputs = Input(shape=(input_dim,))
    # Add a dense (fully connected) layer with 128 neurons.
    x = Dense(128, activation='relu')(inputs)
    # Dropout randomly turns off 30% of neurons to prevent overfitting.
    x = Dropout(0.3)(x)
    # Batch normalization keeps numbers balanced.
    x = BatchNormalization()(x)
    # Classification head: predicts yes/no using two outputs.
    classification = Dense(2, activation='softmax', name='classification')(x)
    # Regression head: predicts a number (like time to event).
    regression = Dense(1, activation='linear', name='regression')(x)
    # Create the model with the two outputs.
    model = Model(inputs=inputs, outputs=[classification, regression])
    # Compile the model with Adam optimizer and proper loss functions.
    model.compile(optimizer=Adam(), 
                  loss={'classification': 'categorical_crossentropy', 'regression': 'mse'},
                  metrics={'classification': 'accuracy', 'regression': 'mae'})
    return model

def build_multimodal_model(input_dim: int) -> tf.keras.Model:
    """
    Build a multimodal model that handles numerical, time-series, image, and genomic data.
    
    Args:
        input_dim (int): The number of features for numerical data.
        
    Returns:
        Model: A compiled Keras model.
    """
    # Numerical branch (for regular numbers)
    input_num = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_num)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Time-series branch (for data over time)
    input_ts = Input(shape=(None, input_dim))
    ts = LSTM(64, return_sequences=False)(input_ts)
    
    # Image branch (for picture data)
    input_img = Input(shape=(64, 64, 3))
    img = Conv2D(32, (3, 3), activation='relu')(input_img)
    img = Flatten()(img)
    
    # Genomic branch (for gene data, here treated like numbers)
    input_genomic = Input(shape=(input_dim,))
    genomic = Dense(64, activation='relu')(input_genomic)
    
    # Merge (or "fuse") all branches together.
    merged = Concatenate()([x, ts, img, genomic])
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.4)(merged)
    
    # Two prediction heads: one for classification and one for regression.
    classification = Dense(2, activation='softmax', name='classification')(merged)
    regression = Dense(1, activation='linear', name='regression')(merged)
    
    model = Model(inputs=[input_num, input_ts, input_img, input_genomic],
                  outputs=[classification, regression])
    model.compile(optimizer=Adam(), 
                  loss={'classification': 'categorical_crossentropy', 'regression': 'mse'},
                  metrics={'classification': 'accuracy', 'regression': 'mae'})
    return model

def build_baseline_model(input_dim: int) -> tf.keras.Model:
    """
    Build a simple model for binary classification to serve as a baseline.
    
    Args:
        input_dim (int): The number of input features.
        
    Returns:
        Model: A compiled Keras Sequential model.
    """
    model = Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid outputs a value between 0 and 1.
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_model_and_scaler(model_path: str = "multimodal_model.h5", scaler_path: str = "scaler.pkl"):
    """
    Load a saved model and scaler from disk so we can use them later.
    
    Args:
        model_path (str): Where the saved model is.
        scaler_path (str): Where the saved scaler is.
        
    model = keras.models.load_model(model_path)
        tuple: The model and scaler objects.
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

#######################################
#       EVALUATION FUNCTIONS          #
#######################################

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model by comparing its predictions to the true labels. This function supports both models that output one channel (baselines binary models and models that output two channels (one-hot encoded classification).
    
    Args:
        model: The Keras model to evaluate.
        X_test (np.ndarray): The test data inputs.
        y_test (np.ndarray): The true labels (if one-hot, then one-hot encoded).
        
    Returns:
        dict: A set of evaluation scores or evaluation metrics.
    """
    # Predict using the model and turn probabilities into 0 or 1.
    #y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    # Get predictions (model.predict returns a list: [classification, regression])
    preds = model.predict(X_test)
     # Determine if model returns a list (multimodal with classification and regression)
    if isinstance(preds, list):
        class_pred = preds[0]
        reg_pred = preds[1]
    else:
        class_pred = preds
        reg_pred = None

    # Check classification output shape.
    # Use the classification branch targets.
    y_test_class = y_test['classification'] if isinstance(y_test, dict) else y_test
    # Determine if the model is a baseline (single-channel) or multimodal (two-channel).
    if class_pred.ndim == 1 or (class_pred.ndim == 2 and class_pred.shape[1] == 1):
        # Single-probability output (baseline model)
        y_pred = (class_pred > 0.5).astype(int)
        if hasattr(y_test_class, "to_numpy"):
            y_true = y_test_class.to_numpy()
        else:
            y_true = y_test_class
        roc_auc = roc_auc_score(y_true, class_pred)
        
    else:
        # Two-channel (one-hot) outputs (multimodal model)
        y_pred = np.argmax(class_pred, axis=1)
        y_true = np.argmax(y_test_class, axis=1)
        #Use positive class probabilities for ROC AUC.
        roc_auc = roc_auc_score(y_true, class_pred[:, 1])
    # Calculate evaluation metrics.
        
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }
    
    if reg_pred is not None and isinstance(y_test, dict) and "regression" in y_test:
        # Use the regression branch targets.
        #Convert to 1D arrays for regression metrics. NumPy's to_numpy() method flattens the array.
        if hasattr(y_test['regression'], "to_numpy"):
            y_test_reg = y_test['regression'].to_numpy().flatten()
        else:
            y_test_reg = np.asarray(y_test['regression']).flatten()
        reg_pred = np.asarray(reg_pred).flatten()
        eps = np.finfo(np.float64).eps
        results.update({
            "mae": mean_absolute_error(y_test_reg, reg_pred),
            "mse": mean_squared_error(y_test_reg, reg_pred),
            "rmse": np.sqrt(mean_squared_error(y_test_reg, reg_pred)),
            "r2": r2_score(y_test_reg, reg_pred),
            "mape": np.mean(np.abs((np.ravel(y_test_reg) - np.ravel(reg_pred)) / np.where(np.ravel(y_test_reg) == 0, eps, np.ravel(y_test_reg)))) * 100
           # "mape": np.mean(np.abs((y_test_reg - reg_pred) / y_test_reg)) * 100
        })
    return results

def ablation_study(multimodal_model, data_dict, target_class_encoded, target_reg, split_func):
    """
    Test the importance of each data type by removing one at a time and evaluating the model.
    
    Args:
        multimodal_model: The model to test.
        data_dict (dict): A dictionary with keys like 'ehr', 'imaging', etc.
        target_class (pd.Series): The true labels.
        split_func: A function to split data into training and test sets.
        
    Returns:
        dict: The evaluation scores for each ablation.
    """
    ablation_results = {}
    
    # Combine all modalities first.
    combined_data = pd.concat([data_dict['ehr'], data_dict['imaging'],
                               data_dict['wearables'], data_dict['genomic']], axis=1)
    #Convert all columns to numeric and fill missing values with 0 (non-convertible items become NaN) and fill NaNs with 0
    combined_data = combined_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Use the encoded targets in the split. Split the combined data along with both targets
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(combined_data, target_class_encoded, target_reg, test_size=0.2, random_state=42)
   
    # Create target dictionaries for training and evaluation.
    y_train = {'classification': y_train_class, 'regression': y_train_reg}
    y_test = {'classification': y_test_class, 'regression': y_test_reg}
    # #Prepare the inputs for the model.//For full data, rebuild a temporary model with the proper input dimensions.
    
    X_train_np = X_train.to_numpy() # Convert to NumPy array. or np.array(X_train)
    input_dim = X_train_np.shape[1]
    
    #Rebuild a new model for the ablation study.
    temp_model = build_multimodal_model(input_dim)
    
    #Prepare the inputs for time-series and image training, including placeholders.
    X_train_ts = X_train_np.reshape(-1, 1, input_dim)
    X_train_img = np.random.rand(len(X_train_np), 64, 64, 3)  # placeholder for image branch
    temp_model.fit( 
        [X_train_np, X_train_ts, X_train_img, X_train_np],
        y_train,
        #{'classification': y_train_class, 'regression': y_train_reg},
        epochs=10,
        batch_size=32,
        verbose=0
    )
    #Prepare inputs for evaluation.
    X_test_np = X_test.to_numpy()
    X_test_ts = X_test_np.reshape(-1, 1, input_dim)
    X_test_img = np.random.rand(len(X_test_np), 64, 64, 3)
    ablation_results['All'] = evaluate_model(
        temp_model, 
        [X_test_np, X_test_ts, X_test_img, X_test_np], 
        y_test
       # {'classification': y_test_class, 'regression': y_test_reg}
    )
    
    # Now evaluate by removing each modality one by one.//ablate one modality at a time
    for modality in ['ehr', 'imaging', 'wearables', 'genomic']:
        remaining = [data for key, data in data_dict.items() if key != modality]
        ablation_data = pd.concat(remaining, axis=1)
        ablation_data = ablation_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_train_sub, X_test_sub, y_train_sub_class, y_test_sub_class, y_train_sub_reg, y_test_sub_reg = train_test_split(ablation_data, target_class_encoded, target_reg, test_size=0.2, random_state=42)
        X_train_sub_np = X_train_sub.to_numpy()
        sub_input_dim = X_train_sub_np.shape[1]
        
        #Build target dictionaries for the ablated dataset
        y_train_sub = {'classification': y_train_sub_class, 'regression': y_train_sub_reg}
        y_test_sub = {'classification': y_test_sub_class, 'regression': y_test_sub_reg}
        
        X_train_sub_np = X_train_sub.to_numpy()
        sub_input_dim = X_train_sub_np.shape[1]
        
        # Rebuild a separate model for this ablation.
        sub_model = build_multimodal_model(sub_input_dim)
        X_train_sub_ts = X_train_sub_np.reshape(-1, 1, sub_input_dim)
        X_train_sub_img = np.random.rand(len(X_train_sub_np), 64, 64, 3)
        sub_model.fit(
            [X_train_sub_np, X_train_sub_ts, X_train_sub_img, X_train_sub_np],
            y_train_sub,
            #{'classification': y_train_sub_class, 'regression': y_train_sub_reg},
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        X_test_sub_np = X_test_sub.to_numpy()
        X_test_sub_ts = X_test_sub_np.reshape(-1, 1, sub_input_dim)
        X_test_sub_img = np.random.rand(len(X_test_sub_np), 64, 64, 3)
        ablation_results[f'No {modality.capitalize()}'] = evaluate_model(
            sub_model, 
            [X_test_sub_np, X_test_sub_ts, X_test_sub_img, X_test_sub_np],
            y_test_sub
           # {'classification': y_test_sub_class, 'regression': y_test_sub_reg}
        )
    
    return ablation_results

def split_data(features, target, test_size=0.2, random_state=42):
    """
    Split the data into training and testing parts.
    
    Args:
        features (np.ndarray or pd.DataFrame): The feature data.
        target (np.ndarray or pd.Series): The target labels.
        
    Returns:
        tuple: The split data (X_train, X_test, y_train, y_test).
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

#######################################
#             API SECTION             #
#######################################
# Define a Pydantic model to validate incoming API data.
class PatientInput(BaseModel):
    Patient_ID: str
    ehr_features: dict
    imaging_features: list
    wearables_data: dict
    genomic_features: list

# Initialize Flask app for our API.
app = Flask(__name__)

def authorize_request(token: str) -> bool:
    """
    Check if the request has a valid token.
    
    Args:
        token (str): The token provided by the requester.
        
    Returns:
        bool: True if authorized, False otherwise.
    """
    # In a real application, talk to an authorization server.
    auth_server = os.environ.get('AUTH_SERVER', 'https://hospital-auth-server.com/validate')
    response = requests.post(auth_server, json={'token': token})
    return response.status_code == 200

# Middleware: Check authorization before every request.
@app.before_request
def middleware():
    token = request.headers.get('Authorization')
    if not token or not authorize_request(token):
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/predict', methods=['POST'])
def predict_cvd():
    """
    API endpoint that takes in patient data and returns a heart disease prediction.
    """
    try:
        # Validate incoming data using our checklist (PatientInput).
        validated_data = PatientInput.parse_obj(request.json)
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400

    data = validated_data.dict()

    try:
        # Convert incoming data into proper formats (tables and arrays).
        ehr_features = pd.DataFrame([data['ehr_features']])
        imaging_features = np.array(data['imaging_features']).reshape(1, -1)
        wearables_df = pd.DataFrame([data['wearables_data']])
        genomic_features = np.array(data['genomic_features']).reshape(1, -1)
        
        # Extract extra features from the wearables (time series).
        time_series_features = extract_time_series_features(wearables_df)
            
        # Combine all features together.
        combined_features = pd.concat([ehr_features, time_series_features], axis=1)
        combined_features = pd.concat([
            combined_features,
            pd.DataFrame(imaging_features),
            pd.DataFrame(genomic_features)
        ], axis=1)
            
        # Scale the combined features using our pre-saved scaler.
        combined_features_scaled = scaler.transform(combined_features)
            
        # Make predictions with the model. (Here we use placeholder arrays for branches like images.)
        predictions = model.predict([
            combined_features_scaled, 
            combined_features_scaled, 
            np.random.rand(len(combined_features_scaled), 64, 64, 3), 
            combined_features_scaled
        ])
        classification = predictions[0]
        regression = predictions[1]
            
        # Interpret the predictions.
        CVD_Presence_prob = classification[0][1]
        CVD_Presence_pred = "Positive" if CVD_Presence_prob > 0.5 else "Negative"
        Time_To_Event = regression[0][0]
            
        # Prepare the response.
        response = {
            "Patient_ID": data['Patient_ID'],
            "CVD_Presence": CVD_Presence_pred,
            "CVD_Presence_probability": CVD_Presence_prob,
            "Time_To_Event": Time_To_Event
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """
    A simple test endpoint to show an example of patient data.
    """
    sample_data = {
        "Patient_ID": "12345",
        "ehr_features": {
            "age": 55,
            "sex": 1,
            "cp": 3,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.5,
            "slope": 2,
            "ca": 1,
            "thal": 3
        },
        "imaging_features": [0.1, 0.3, 0.5, 0.7, 0.9],
        "wearables_data": {
            "Patient_ID": "12345",
            "steps_per_day": 8000,
            "avg_heart_rate": 75,
            "sleep_hours": 7,
            "activity_level": "Moderate",
            "time": 1  # a placeholder value needed for tsfresh extraction
        },
        "genomic_features": [1, 0, 2, 1, 1, 0, 1, 2, 2, 0]
    }
    return jsonify(sample_data)

#######################################
#           MAIN TRAINING CODE        #
#######################################

def run_training():
    """
    Load data, train the multimodal model, evaluate it, and save the model and scaler.
    """
    # File names for our data (adjust paths as needed).
    EHR_FILE = r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/ehr_data.csv'
    IMAGING_FILE = r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/imaging_data.csv'
    WEARABLES_FILE = r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/wearables_data.csv'
    GENOMIC_FILE = r'/Users/manny/Desktop/Desktop Folder/MultimodalDataset021525/genomic_data.csv'
    
    # Load data using Spark and then convert to Pandas.
    ehr_data = load_data_spark(EHR_FILE).toPandas()
    imaging_data = load_data_spark(IMAGING_FILE).toPandas()
    wearables_data = load_data_spark(WEARABLES_FILE).toPandas()
    genomic_data = load_data_spark(GENOMIC_FILE).toPandas()
    
    # Fill in missing numbers and show missing data plots.
    ehr_data = impute_and_visualize(ehr_data)
    imaging_data = impute_and_visualize(imaging_data)
    wearables_data = impute_and_visualize(wearables_data)
    genomic_data = impute_and_visualize(genomic_data)
    
    # Combine the different data tables into one big table.
    merged_data = merge_data(ehr_data, imaging_data, wearables_data, genomic_data)
    
    # Extract extra features from the wearables time series and merge them.
    
    time_series_features = extract_time_series_features(wearables_data).reset_index()
    if 'Patient_ID' not in time_series_features.columns:
        time_series_features = time_series_features.rename(columns={'index': 'Patient_ID'})
    merged_data = merged_data.merge(time_series_features, on='Patient_ID')
    
    # Ensure "Time_To_Event" column exists. If it doesn't, create a dummy column.
    if 'Time_To_Event' not in merged_data.columns:
        print("Warning: 'Time_To_Event' column not found. Assigning a default value (0) to this column.")
        merged_data['Time_To_Event'] = 0
    
    # Prepare our features by removing unwanted columns.
    features = merged_data.drop(columns=['Patient_ID', 'CVD_Presence', 'Time_To_Event'], errors='ignore')
    scaler_local = StandardScaler()
    features_scaled = scaler_local.fit_transform(features)  # Scale the features once.
    
    features_scaled = np.nan_to_num(features_scaled)
    
    print("Merged data columns:", merged_data.columns.tolist())
    # Get the labels we want to predict.
    target_class = merged_data['CVD_Presence']
    
    
    target_reg = merged_data['Time_To_Event'] # Time to event (regression) is a number. 
    
    target_class_encoded = to_categorical(target_class)
    
    # Split the data into training and testing sets.
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        features_scaled, target_class_encoded, target_reg, test_size=0.2, random_state=42
    )
    
    # Build the multimodal model using the number of features.
    input_dim = features.shape[1]
    multimodal_model = build_multimodal_model(input_dim)
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
     # Reshape features_scaled for the time-series branch: add a time step dimension.
    features_scaled_ts = features_scaled.reshape(-1, 1, input_dim)
    
    
    # Train the model. (For the time-series and image parts, we use placeholder data.)
    history = multimodal_model.fit(
        [features_scaled, features_scaled_ts, np.random.rand(len(features_scaled), 64, 64, 3), features_scaled],
        {'classification': target_class_encoded, 'regression': target_reg},
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Evaluate the model on the test set.
    X_test_ts = X_test.reshape(-1, 1, input_dim)
    test_loss, cls_loss, reg_loss, test_accuracy, test_mae = multimodal_model.evaluate(
        [X_test, X_test_ts, np.random.rand(len(X_test), 64, 64, 3), X_test],
        {'classification': y_test_class, 'regression': y_test_reg}
    )
    print(f"Test Loss: {test_loss},cls_loss: {cls_loss}, reg_loss: {reg_loss}, Test Accuracy: {test_accuracy}, Test MAE: {test_mae}")
    
    # Calculate classification metrics.
    #y_class_pred = np.argmax(multimodal_model.predict(X_test)[0], axis=1)
    # Reshape X_test for the time-series branch.
    X_test_ts = X_test.reshape(-1, 1, input_dim)
# Use placeholder image data for the image branch.
    X_test_img = np.random.rand(len(X_test), 64, 64, 3)
# Use X_test again for the genomic branch.
    predictions = multimodal_model.predict([X_test, X_test_ts, X_test_img, X_test])
    y_class_pred = np.argmax(predictions[0], axis=1)
    y_class_true = np.argmax(y_test_class, axis=1)
    accuracy = accuracy_score(y_class_true, y_class_pred)
    precision = precision_score(y_class_true, y_class_pred)
    recall = recall_score(y_class_true, y_class_pred)
    f1 = f1_score(y_class_true, y_class_pred)
    roc_auc = roc_auc_score(y_test_class, predictions[0])
    print(f"Classification Metrics:\n"
          f"Accuracy: {accuracy:.20f}\n"
          f"Precision: {precision:.20f}\n"
          f"Recall: {recall:.20f}\n"
          f"F1-score: {f1:.20f}\n"
          f"AUC-ROC: {roc_auc:.20f}")
    
# Calculate regression metrics.
# Prepare the inputs for all branches:
    X_test_ts = X_test.reshape(-1, 1, input_dim)
    X_test_img = np.random.rand(len(X_test), 64, 64, 3)
# Predict using all 4 inputs:
    preds = multimodal_model.predict([X_test, X_test_ts, X_test_img, X_test])
    y_reg_pred = preds[1]

   # y_reg_pred = multimodal_model.predict(X_test)[1]
    mae = mean_absolute_error(y_test_reg, y_reg_pred)
    mse = mean_squared_error(y_test_reg, y_reg_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_reg_pred)
    eps = np.finfo(np.float64).eps
    mape = np.mean(np.abs((np.ravel(y_test_reg) - np.ravel(y_reg_pred)) / 
                      np.where(np.ravel(y_test_reg)==0, eps, np.ravel(y_test_reg)))) * 100
    #mape = np.mean(np.abs((np.ravel(y_test_reg) - np.ravel(y_reg_pred)) / np.ravel(y_test_reg))) * 100
    print(f"Regression Metrics:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR-squared: {r2}\nMAPE: {mape}")
    
    # Build and evaluate a simple baseline model (using EHR data only as an example).
    X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
    ehr_data.drop(columns=['Patient_ID']), target_class, test_size=0.2, random_state=42
    )
    baseline_model = build_baseline_model(X_train_baseline.shape[1])
    baseline_model.fit(X_train_baseline, y_train_baseline, epochs=10, batch_size=32, verbose=0)
    baseline_results = evaluate_model(baseline_model, X_test_baseline, y_test_baseline)
    print("Baseline EHR Model Results:", baseline_results)
    
    # Perform ablation studies to see which data parts are most important.
    data_dict = {
        'ehr': ehr_data,
        'imaging': imaging_data,
        'wearables': wearables_data,
        'genomic': genomic_data
    }
    ablation_results = ablation_study(multimodal_model, data_dict, target_class_encoded, target_reg, split_data)
    print("Ablation Study Results:", ablation_results)
    
    def wrapped_predict(x):
    # x is a 2D array with shape (n_samples, feature_dim)
        x_ts = x.reshape(-1, 1, x.shape[1])
    # Use a placeholder for image branch (random images)
        x_img = np.random.rand(x.shape[0], 64, 64, 3)
    # Use x again for the genomic branch (or adjust as needed)
        preds = multimodal_model.predict([x, x_ts, x_img, x])
    # Return classification predictions (or both outputs if needed)
    return preds[0]
    
    # Use SHAP to explain model predictions (make a summary plot).
    explainer = shap.KernelExplainer(wrapped_predict, features_scaled)
    shap_values = explainer.shap_values(features_scaled)
    shap.summary_plot(shap_values, features_scaled)
    import matplotlib.pyplot as plt
    plt.show()
    # Use SHAP to explain model predictions (make a force plot for the first prediction).
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0], features_scaled[0])
    # Use SHAP to explain model predictions (make a decision plot for the first prediction).
    shap.decision_plot(explainer.expected_value, shap_values[0], features_scaled[0])
# Save the trained model and the scaler for later use.
 #def get_model_and_scaler(model_path: str = None, scaler_path: str = None):
    import os

# Determine the directory where multimodalcardio.py is located.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full paths for saving the model and scaler.
    model_save_path = os.path.join(base_dir, "multimodal_model.h5")
    scaler_save_path = os.path.join(base_dir, "scaler.pkl")

# Save the trained model and the scaler to these paths.
    multimodal_model.save(model_save_path)
    joblib.dump(scaler_local, scaler_save_path)
    print("Model and scaler saved to disk at:")
    print("Model path:", model_save_path)
    print("Scaler path:", scaler_save_path)

#######################################
#             MAIN BLOCK              #
#######################################
if __name__ == '__main__':
    # Check the command-line arguments to decide what to run.
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "train":
            print("Running training and evaluation...")
            run_training()
        elif mode == "api":
            print("Starting the API server on port 5000...")
            # When running the API, we need the model and scaler to be available.
            # We assume they have been saved already; if not, run training first.
            try:
                model, scaler = get_model_and_scaler()
            except Exception as e:
                print("Error loading model/scaler. Please run training first. Error:", e)
                sys.exit(1)
            app.run(debug=True, port=5000)
        else:
            print("Usage: python multimodalcardio.py [train|api]")
    else:
        print("Usage: python multimodalcardio.py [train|api]")
