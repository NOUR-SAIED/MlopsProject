# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import logging
from mlProject.pipeline.prediction import PredictionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
try:
    logger.info("🚀 Starting Flask app, loading model...")
    model = PredictionPipeline()
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Kubernetes"""
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "model": "loaded"}), 200

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return "Model not loaded. Please check server logs.", 503
        
        # Get form data
        input_data = {k: request.form[k] for k in request.form.keys()}
        
        # Convert numeric columns
        numeric_cols = [
            "Age","Membership_Years","Login_Frequency","Session_Duration_Avg",
            "Pages_Per_Session","Cart_Abandonment_Rate","Wishlist_Items",
            "Total_Purchases","Average_Order_Value","Days_Since_Last_Purchase",
            "Discount_Usage_Rate","Returns_Rate","Email_Open_Rate",
            "Customer_Service_Calls","Product_Reviews_Written",
            "Social_Media_Engagement_Score","Mobile_App_Usage",
            "Payment_Method_Diversity","Lifetime_Value","Credit_Balance"
        ]
        
        for key in numeric_cols:
            if key in input_data:
                try:
                    input_data[key] = float(input_data[key])
                except ValueError:
                    return f"Invalid value for {key}. Must be a number.", 400
        
        # Convert to DataFrame and predict
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        
        logger.info(f"📊 Prediction made: {prediction}")
        return render_template("results.html", prediction=prediction)
        
    except KeyError as e:
        return f"Missing field: {e}", 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Error making prediction: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)