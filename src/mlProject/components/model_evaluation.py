import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject import logger
from sklearn.pipeline import Pipeline
from mlProject.components.data_transformation import Preprocessor

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # 🔹 Définir les variables d'environnement pour DagsHub
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = "hannamhiri"  
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "d818c76624661ed3e44ed5cd15bb08d17cd94c4d"

    def eval_metrics(self, actual, pred, prob=None):
        """
        actual : vraies valeurs
        pred : prédictions binaires
        prob : probabilité pour calcul ROC-AUC (optionnel)
        """
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)
        roc_auc = roc_auc_score(actual, prob) if prob is not None else None

        return accuracy, precision, recall, f1, roc_auc

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

        for model_file in os.listdir(self.config.model_dir):
            if not model_file.endswith(".pkl"):
                continue

            model_name = model_file.replace(".pkl", "")
            model_path = os.path.join(self.config.model_dir, model_file)

            logger.info(f"Evaluating model: {model_name}")

            with mlflow.start_run(run_name=model_name):
                try:
                    # --- NEW: Load the preprocessor created in this run ---
                    # 1. Construct the path to the preprocessor saved by DataTransformation
                    preprocessor_path = os.path.join('artifacts/data_transformation/preprocessor.pkl')
                    
                    # 2. Load the dictionary and recreate the Preprocessor object
                    preprocessor_dict = joblib.load(preprocessor_path)
                    
                    inference_preprocessor = Preprocessor(
                        num_cols=preprocessor_dict["num_cols"],
                        cat_cols=preprocessor_dict["cat_cols"],
                        drop_cols=preprocessor_dict.get("drop_cols", [])
                    )
                    inference_preprocessor.label_encoders = preprocessor_dict["label_encoders"]
                    logger.info(f"✅ Loaded preprocessor for {model_name}")

                    # 3. Load the trained model
                    model = joblib.load(model_path)

                    # --- NEW: Create a single deployable Pipeline ---
                    deployable_pipeline = Pipeline(steps=[
                        ('preprocessor', inference_preprocessor),  # Step 1: Apply encoding
                        ('classifier', model)                       # Step 2: Make prediction
                    ])
                    
                    # --- UPDATED: Use the PIPELINE for evaluation ---
                    predictions = deployable_pipeline.predict(test_x)
                    if hasattr(deployable_pipeline, "predict_proba"):
                        probas = deployable_pipeline.predict_proba(test_x)[:, 1]
                    else:
                        probas = None

                except FileNotFoundError as e:
                    logger.error(f"❌ Missing preprocessor file: {e}")
                    logger.warning("⚠️ Falling back to raw model evaluation (no preprocessing)")
                    
                    # Fallback: load and use raw model
                    model = joblib.load(model_path)
                    deployable_pipeline = model  # For backward compatibility
                    
                    predictions = model.predict(test_x)
                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(test_x)[:, 1]
                    else:
                        probas = None

                # === KEEP YOUR EXISTING METRICS AND LOGGING CODE ===
                accuracy, precision, recall, f1, roc_auc = self.eval_metrics(test_y, predictions, probas)

                scores = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                }

                save_json(
                    path=Path(os.path.join(self.config.root_dir, f"{model_name}_metrics.json")),
                    data=scores
                )

                # Log params if available
                if hasattr(self.config, "all_params") and model_name in self.config.all_params:
                    mlflow.log_params(self.config.all_params[model_name])

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)

                # --- CRITICAL CHANGE: Log the PIPELINE to MLflow ---
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        deployable_pipeline,        # Log the complete pipeline (or raw model as fallback)
                        "model",
                        registered_model_name=f"{model_name}Model"
                    )
                    logger.info(f"✅ Logged {'pipeline' if isinstance(deployable_pipeline, Pipeline) else 'model'} for {model_name} to MLflow")
                else:
                    mlflow.sklearn.log_model(deployable_pipeline, "model")