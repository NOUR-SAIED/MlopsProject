# test_transformation.py
import pandas as pd
from src.mlProject.components.data_transformation import DataTransformation
from mlProject.entity.config_entity import DataTransformationConfig

print("Testing DataTransformation...")

# Create config
config = DataTransformationConfig(
    root_dir='artifacts/data_transformation',
    data_path='artifacts/data_ingestion/weatherAUS.csv'  # Check this path!
)

# Check if source data exists
import os
print(f"Source data exists: {os.path.exists(config.data_path)}")
print(f"Source data path: {config.data_path}")

# List what's in data_ingestion
print("\nFiles in artifacts/data_ingestion/:")
for item in os.listdir('artifacts/data_ingestion'):
    print(f"  - {item}")