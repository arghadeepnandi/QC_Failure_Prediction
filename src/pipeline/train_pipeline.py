import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    """
    Complete training pipeline orchestrating all components
    """
    def __init__(self):
        pass
    
    def run_training_pipeline(self):
        """
        Execute the complete training pipeline
        """
        try:
            logging.info("=" * 50)
            logging.info("Training Pipeline Started")
            logging.info("=" * 50)
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info("=" * 50)
            logging.info(f"Training Pipeline Completed Successfully!")
            logging.info(f"Final Model Accuracy: {accuracy:.4f}")
            logging.info("=" * 50)
            
            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run the training pipeline
    pipeline = TrainPipeline()
    pipeline.run_training_pipeline()