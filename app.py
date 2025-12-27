from flask import Flask, render_template, request
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
pipeline = PredictionPipeline()  # Load once at startup - FIXED: renamed to 'pipeline' to be consistent

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Build dict from form
        input_data = {k: request.form[k] for k in request.form}

        # 2. Convert numeric fields (critical for preprocessing)
        numeric_cols = [
            "Age", "Membership_Years", "Login_Frequency", "Session_Duration_Avg",
            "Pages_Per_Session", "Cart_Abandonment_Rate", "Wishlist_Items",
            "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase",
            "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
            "Customer_Service_Calls", "Product_Reviews_Written",
            "Social_Media_Engagement_Score", "Mobile_App_Usage",
            "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance"
        ]
        for col in numeric_cols:
            if col in input_data:
                input_data[col] = float(input_data[col])

        # ✅ 3. CONVERT TO DATAFRAME (1 row) — THIS IS THE KEY FIX
        df = pd.DataFrame([input_data])  # ← list of dict → (1, N) DataFrame

        # ✅ 4. Now safe to call predict() — USING THE GLOBAL PIPELINE INSTANCE
        pred = pipeline.predict(df)[0]  # FIXED: Using the global 'pipeline' instead of creating new one

        return render_template("results.html", prediction=int(pred))

    except Exception as e:
        return f"❌ Prediction failed: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)