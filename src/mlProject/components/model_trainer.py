import pandas as pd
import os
from mlProject import logger
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from mlProject.entity.config_entity import ModelTrainerConfig
# Modèles
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class SMOTEHandler:
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def fit_resample(self, X_train, y_train):
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, all_params: dict):
        self.config = config
        self.all_params = all_params  # dictionnaire contenant les hyperparamètres pour chaque modèle
        self.smote_handler = SMOTEHandler(random_state=42)  # ajout du handler SMOTE

    def train(self):
        # 1️⃣ Charger les données
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # 2️⃣ Séparer X / y
        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]

        # 3️⃣ Appliquer SMOTE via SMOTEHandler uniquement sur le train
        X_train_res, y_train_res = self.smote_handler.fit_resample(X_train, y_train)

        logger.info("Target distribution AFTER SMOTE:")
        logger.info(pd.Series(y_train_res).value_counts().to_string())
        logger.info(f"Train before SMOTE: {X_train.shape}")
        logger.info(f"Train after SMOTE: {X_train_res.shape}")

        # 4️⃣ Définir tous les modèles avec leurs paramètres depuis all_params
        baseline_models = {
            'RandomForest': RandomForestClassifier(
                **self.all_params.get('RandomForest', {}),
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                **self.all_params.get('XGBoost', {}),
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                **self.all_params.get('LightGBM', {}),
                random_state=42
            ),
            'CatBoost': CatBoostClassifier(
                **self.all_params.get('CatBoost', {}),
                random_seed=42,
                verbose=0
            )
        }

        # 5️⃣ Boucler sur tous les modèles pour entraîner et sauvegarder
        os.makedirs(self.config.root_dir, exist_ok=True)
        for name, model in baseline_models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_res, y_train_res)
            model_path = os.path.join(self.config.root_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"{name} saved at {model_path}")




