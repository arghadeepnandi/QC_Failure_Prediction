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

    def predict(self, features: pd.DataFrame):
        """
        Make predictions using the trained model and preprocessor
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor for prediction")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # =========================
            # 🔑 FIX: Handle missing sensor columns
            # =========================
            expected_columns = preprocessor.feature_names_in_

            for col in expected_columns:
                if col not in features.columns:
                    features[col] = np.nan

            # Ensure correct column order
            features = features[expected_columns]

            logging.info(f"Final prediction input shape: {features.shape}")

            # Transform features
            data_preprocessed = preprocessor.transform(features)

            # Predict
            preds = model.predict(data_preprocessed)
            pred_proba = model.predict_proba(data_preprocessed)

            pred_labels = ['PASS' if pred == 0 else 'FAIL' for pred in preds]

            logging.info(f"Prediction completed: {pred_labels}")

            return pred_labels, pred_proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom input data for prediction
    """

    def __init__(self, sensor_values: dict):
        self.sensor_values = sensor_values

    def get_data_as_data_frame(self):
        try:
            df = pd.DataFrame([self.sensor_values])
            logging.info(f"Created input DataFrame with {df.shape[1]} sensors")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def from_csv_row(csv_path: str, row_index: int = 0):
        try:
            df = pd.read_csv(csv_path)

            row = df.iloc[row_index]
            if 'QC_Failure' in row.index:
                row = row.drop('QC_Failure')

            return CustomData(row.to_dict())

        except Exception as e:
            raise CustomException(e, sys)
