import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Class for training and evaluating multiple models
    Each model is wrapped in a pipeline with StandardScaler
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models and select the best one
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            logging.info(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            
            # Define models - each wrapped in pipeline with StandardScaler
            # Using class_weight='balanced' for handling imbalanced data
            models = {
                "Logistic Regression": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(
                        max_iter=1000, 
                        class_weight='balanced',
                        random_state=42
                    ))
                ]),
                "Random Forest": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(
                        class_weight='balanced',
                        random_state=42
                    ))
                ]),
                "Decision Tree": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', DecisionTreeClassifier(
                        class_weight='balanced',
                        random_state=42
                    ))
                ]),
                "Gradient Boosting": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', GradientBoostingClassifier(random_state=42))
                ]),
                "SVM": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(
                        class_weight='balanced',
                        random_state=42,
                        probability=True
                    ))
                ]),
                "Naive Bayes": Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', GaussianNB())
                ])
            }
            
            # Define hyperparameters for tuning
            params = {
                "Logistic Regression": {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l2']
                },
                "Random Forest": {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5]
                },
                "Decision Tree": {
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__max_depth': [5, 10, 15, None],
                    'classifier__min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__n_estimators': [50, 100, 150],
                    'classifier__max_depth': [3, 5, 7]
                },
                "SVM": {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__kernel': ['rbf', 'linear']
                },
                "Naive Bayes": {}
            }
            
            # Evaluate all models
            logging.info("Starting model evaluation with hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            
            # Get best model score from dict
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]['test_accuracy']
            )
            best_model_score = model_report[best_model_name]['test_accuracy']
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with accuracy > 0.6", sys)
            
            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Best model test accuracy: {best_model_score:.4f}")
            logging.info(f"Best model F1-score: {model_report[best_model_name]['f1_score']:.4f}")
            
            # Retrain best model on full training data
            best_model.fit(X_train, y_train)
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Final prediction and evaluation
            predicted = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, predicted)
            
            logging.info("=" * 60)
            logging.info("Final Classification Report:")
            logging.info("\n" + classification_report(y_test, predicted))
            logging.info("=" * 60)
            
            logging.info(f"Model training completed. Final accuracy: {final_accuracy:.4f}")
            
            return final_accuracy
        
        except Exception as e:
            raise CustomException(e, sys)