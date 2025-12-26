# src/mlProject/pipeline/prediction.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        logger.info("Loading model pipeline from MLflow 'Staging' stage...")
        
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", 
                                        "https://dagshub.com/hannamhiri/MlopsProject.mlflow"))
        
        try:
            client = MlflowClient()
            
            # Find model in Staging
            models = client.search_registered_models()
            model_info = None
            for rm in models:
                for version in rm.latest_versions:
                    if version.current_stage == "Staging":
                        model_info = {"name": rm.name, "version": version.version}
                        logger.info(f"Found {rm.name} v{version.version} in Staging")
                        break
                if model_info:
                    break
            
            if not model_info:
                raise ValueError("No model found in 'Staging' stage")
            
            # Load the COMPLETE pipeline from MLflow
            model_uri = f"models:/{model_info['name']}/{model_info['version']}"
            self.pipeline = mlflow.sklearn.load_model(model_uri)
            
            # Check what we loaded
            from sklearn.pipeline import Pipeline
            if isinstance(self.pipeline, Pipeline):
                logger.info("✅ Loaded complete model pipeline (with preprocessing)")
            else:
                logger.info("✅ Loaded raw model (no preprocessing in pipeline)")
            
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
            raise
    
    def predict(self, data: pd.DataFrame):
        """Predict. The pipeline handles all preprocessing automatically."""
        return self.pipeline.predict(data)