import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple models with hyperparameter tuning
    Returns a report with model performance metrics
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Get parameters for the current model
            param = params.get(model_name, {})
            
            if param:
                # Perform grid search for hyperparameter tuning
                gs = GridSearchCV(model, param, cv=3, scoring='accuracy', n_jobs=-1)
                gs.fit(X_train, y_train)
                
                # Set best parameters
                model.set_params(**gs.best_params_)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Additional metrics for binary classification
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # Store results
            report[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logging.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, F1-Score: {f1:.4f}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)