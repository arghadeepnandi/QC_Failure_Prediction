import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    Class for transforming and preprocessing SECOM sensor data
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, numeric_features):
        """
        Create and return preprocessing pipeline
        Only numeric features (all sensor columns)
        """
        try:
            # Numeric pipeline: median imputation only
            # StandardScaler will be applied in the model pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )
            
            logging.info(f"Number of numeric features: {len(numeric_features)}")
            
            # Create preprocessor with only numeric pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_features)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply preprocessing to train and test data
        - Drop high missing columns (>70%)
        - Impute remaining with median
        """
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            # Define target column
            target_column_name = "QC_Failure"
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Identifying high missing columns (>70%)")
            
            # Calculate missing percentage
            missing_percentage = input_feature_train_df.isnull().mean() * 100
            high_missing_cols = missing_percentage[missing_percentage > 70].index.tolist()
            
            logging.info(f"Dropping {len(high_missing_cols)} sensors with >70% missing values")
            
            # Drop high missing columns
            input_feature_train_df = input_feature_train_df.drop(columns=high_missing_cols)
            input_feature_test_df = input_feature_test_df.drop(columns=high_missing_cols)
            
            logging.info(f"Features after dropping: {input_feature_train_df.shape[1]}")
            
            # Get remaining numeric features
            numeric_features = input_feature_train_df.columns.tolist()
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numeric_features)
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Transform features (only imputation)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Get target as numpy arrays
            target_train_arr = target_feature_train_df.values
            target_test_arr = target_feature_test_df.values
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, target_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_test_arr]
            
            logging.info("Saved preprocessing object")
            
            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)