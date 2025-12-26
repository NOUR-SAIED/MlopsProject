import mlflow
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path
import os
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        print("🔄 Loading model and preprocessor...")
        
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/hannamhiri/MlopsProject.mlflow").strip()
        )

        try:
            # 🔹 Load preprocessor
            self.preprocessor = joblib.load(Path("artifacts/data_transformation/preprocessor.pkl"))
            print("✅ Preprocessor loaded")

            # 🔹 Load model from MLflow Staging
            client = MlflowClient()
            model_uri = None
            for rm in client.search_registered_models():
                for mv in rm.latest_versions:
                    if mv.current_stage == "Staging":
                        model_uri = f"models:/{rm.name}/{mv.version}"
                        print(f"✅ Found {rm.name} v{mv.version} in 'Staging'")
                        break
                if model_uri:
                    break

            if not model_uri:
                raise ValueError("No model in 'Staging' stage")

            self.model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"⚠️ MLflow/preprocessor failed: {e}")
            print("Falling back to local LightGBM.pkl + preprocessor.pkl...")
            self.preprocessor = joblib.load(Path("artifacts/data_transformation/preprocessor.pkl"))
            self.model = joblib.load(Path("artifacts/model_trainer/LightGBM.pkl"))
            print("✅ Local model + preprocessor loaded")

    def predict(self, data: pd.DataFrame):  # ✅ Fixed: data: pd.DataFrame
        X_processed = self.preprocessor.transform(data)
        return self.model.predict(X_processed)