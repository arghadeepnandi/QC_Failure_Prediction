import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """
    Pipeline for making predictions on new SECOM sensor data
    """
    def __init__(self):
        pass
    
    def predict(self, features):
        """
        Make predictions using the trained model and preprocessor
        """
        try:
            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info("Loading model and preprocessor for prediction")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform features using preprocessor (imputation)
            logging.info("Transforming input features")
            data_preprocessed = preprocessor.transform(features)
            
            # Make prediction (model pipeline includes StandardScaler)
            logging.info("Making prediction")
            preds = model.predict(data_preprocessed)
            
            # Get prediction probabilities
            pred_proba = model.predict_proba(data_preprocessed)
            
            # Convert prediction to label
            pred_labels = ['PASS' if pred == 0 else 'FAIL' for pred in preds]
            
            logging.info(f"Prediction completed: {pred_labels}")
            logging.info(f"Prediction probabilities: {pred_proba}")
            
            return pred_labels, pred_proba
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom input data for prediction
    Accepts sensor readings from SECOM dataset
    """
    def __init__(self, sensor_values: dict):
        """
        Initialize with dictionary of sensor values
        sensor_values: dict with keys like 'sensor_0', 'sensor_1', etc.
        """
        self.sensor_values = sensor_values
    
    def get_data_as_data_frame(self):
        """
        Convert custom data to pandas DataFrame
        """
        try:
            # Create DataFrame with sensor values
            custom_data_df = pd.DataFrame([self.sensor_values])
            
            logging.info(f"Created input DataFrame with {len(self.sensor_values)} sensors")
            
            return custom_data_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def from_csv_row(csv_path: str, row_index: int = 0):
        """
        Create CustomData from a row in QC.csv
        Useful for testing with existing data
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Get the row
            row = df.iloc[row_index]
            
            # Remove target column if exists
            if 'QC_Failure' in row.index:
                row = row.drop('QC_Failure')
            
            # Convert to dictionary
            sensor_values = row.to_dict()
            
            return CustomData(sensor_values)
        
        except Exception as e:
            raise CustomException(e, sys)