from flask import Flask, request, render_template
import pandas as pd
import json
import io

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException
import sys

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    """
    Render the home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle prediction requests
    Accepts either JSON sensor data or CSV file upload
    """
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            # Check if file upload or manual input
            if 'csv_file' in request.files and request.files['csv_file'].filename != '':
                # File upload method
                file = request.files['csv_file']
                
                logging.info(f"Received CSV file: {file.filename}")
                
                # Read CSV file
                df = pd.read_csv(file)
                
                # Remove target column if exists
                if 'QC_Failure' in df.columns:
                    df = df.drop(columns=['QC_Failure'])
                
                # Take first row for prediction
                sensor_values = df.iloc[0].to_dict()
                
            else:
                # Manual JSON input method
                sensor_data_str = request.form.get('sensor_data')
                
                logging.info("Received manual sensor data")
                
                # Parse JSON string
                sensor_values = json.loads(sensor_data_str)
            
            # Create CustomData object
            data = CustomData(sensor_values=sensor_values)
            
            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            
            logging.info(f"Prediction input shape: {pred_df.shape}")
            logging.info(f"Prediction input columns: {len(pred_df.columns)}")
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results, probabilities = predict_pipeline.predict(pred_df)
            
            # Get the probability for the predicted class
            result = results[0]
            if result == 'PASS':
                prob = probabilities[0][0]  # Probability of class 0 (PASS)
            else:
                prob = probabilities[0][1]  # Probability of class 1 (FAIL)
            
            logging.info(f"Prediction result: {result} (confidence: {prob:.4f})")
            
            return render_template('index.html', results=result, probability=prob)
        
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            return render_template('index.html', error="Invalid JSON format. Please check your input.")
        
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Handle batch predictions from CSV file
    Returns predictions for all rows
    """
    try:
        if 'csv_file' not in request.files:
            return {"error": "No file uploaded"}, 400
        
        file = request.files['csv_file']
        
        if file.filename == '':
            return {"error": "No file selected"}, 400
        
        logging.info(f"Received batch CSV file: {file.filename}")
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Remove target column if exists
        if 'QC_Failure' in df.columns:
            actual_labels = df['QC_Failure'].tolist()
            df = df.drop(columns=['QC_Failure'])
        else:
            actual_labels = None
        
        logging.info(f"Processing {len(df)} samples")
        
        # Make predictions
        predict_pipeline = PredictPipeline()
        results, probabilities = predict_pipeline.predict(df)
        
        # Create response
        predictions = []
        for i, (result, probs) in enumerate(zip(results, probabilities)):
            pred_dict = {
                'row': i,
                'prediction': result,
                'pass_probability': float(probs[0]),
                'fail_probability': float(probs[1])
            }
            if actual_labels:
                pred_dict['actual'] = 'PASS' if actual_labels[i] == 0 else 'FAIL'
            predictions.append(pred_dict)
        
        logging.info(f"Batch prediction completed for {len(predictions)} samples")
        
        return {"predictions": predictions}, 200
    
    except Exception as e:
        logging.error(f"Error during batch prediction: {str(e)}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)