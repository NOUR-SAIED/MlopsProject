import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

class Preprocessor:
    def __init__(self, num_cols, cat_cols, drop_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_encoders = {}
        self.drop_cols = drop_cols if drop_cols is not None else []

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X_train = X_train.copy()
        
        # Supprimer les colonnes inutiles
        X_train = X_train.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = [c for c in self.num_cols if c not in self.drop_cols]
        self.cat_cols = [c for c in self.cat_cols if c not in self.drop_cols]

        # Numérique : imputation + clipping + outliers
        for col in self.num_cols:
            X_train[col] = X_train[col].fillna(X_train[col].median())
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_train[col] = X_train[col].clip(lower, upper)

        # Catégorique : LabelEncoder
        for col in self.cat_cols:
            le = LabelEncoder()
            # If there are missing values or unseen labels this may fail; ensure consistent handling
            X_train[col] = X_train[col].astype(str).fillna("nan")
            X_train[col] = le.fit_transform(X_train[col])
            self.label_encoders[col] = le
        
        return X_train

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test = X_test.copy()

        # Supprimer les colonnes inutiles
        X_test = X_test.drop(columns=self.drop_cols, errors="ignore")

        # Numérique : imputation + clipping + outliers
        for col in self.num_cols:
            X_test[col] = X_test[col].fillna(X_test[col].median())
            Q1 = X_test[col].quantile(0.25)
            Q3 = X_test[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X_test[col] = X_test[col].clip(lower, upper)

        # Catégorique : transformation avec LabelEncoder sauvegardé
        for col in self.cat_cols:
            le = self.label_encoders.get(col)
            if le is None:
                raise KeyError(f"LabelEncoder for column '{col}' not found in preprocessor.")
            X_test[col] = X_test[col].astype(str).fillna("nan")
            X_test[col] = le.transform(X_test[col])
        
        return X_test

    def save(self, filepath: str = "preprocessor.pkl"):
        """
        Save the Preprocessor instance so that joblib.load returns an object
        with a .transform(...) method (used by the PredictionPipeline).
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor instance saved to {filepath}")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, target: str):
        self.config = config
        self.target = target
        self.df = pd.read_csv(self.config.data_path)

    def transform_and_split(self, test_size=0.20, random_state=42):
        # Colonnes numériques et catégoriques
        num_cols = [
            "Age", "Session_Duration_Avg", "Pages_Per_Session", "Wishlist_Items",
            "Days_Since_Last_Purchase", "Discount_Usage_Rate", "Returns_Rate",
            "Email_Open_Rate", "Customer_Service_Calls", "Product_Reviews_Written",
            "Social_Media_Engagement_Score", "Mobile_App_Usage", 
            "Payment_Method_Diversity", "Credit_Balance"
        ]
        cat_cols = ["Gender", "Country", "City", "Signup_Quarter"]
        drop_cols = ["Gender", "Signup_Quarter", "Country"]  # colonnes à supprimer
        # Mettre à jour cat_cols après drop
        cat_cols = [c for c in cat_cols if c not in drop_cols]

        # Split features/target
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Sauvegarder l'ordre des features
        FEATURES = X_train.columns.tolist()
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(FEATURES, os.path.join(self.config.root_dir, "feature_order.pkl"))

        # Preprocessing
        preprocessor = Preprocessor(num_cols, cat_cols, drop_cols=drop_cols)
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        # Save the Preprocessor instance (so PredictionPipeline can load and use it)
        preprocessor_path = os.path.join(self.config.root_dir, "preprocessor.pkl")
        preprocessor.save(preprocessor_path)

        # Ajouter la target correctement
        X_train_prep = X_train_prep.copy()
        X_test_prep = X_test_prep.copy()
        X_train_prep[self.target] = y_train.values
        X_test_prep[self.target] = y_test.values

        # Sauvegarder CSV
        X_train_prep.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        X_test_prep.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data transformed and split into training and test sets")
        print(f"Train shape: {X_train_prep.shape}, Test shape: {X_test_prep.shape}")

        return X_train_prep, X_test_prep, y_train, y_test