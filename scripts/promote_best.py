import json
import os
import sys
import mlflow

def load_metric(model_name, metric_name="roc_auc"):
    """Load the desired metric from the model's local JSON file"""
    metrics_dir = os.path.join("artifacts", "model_evaluation")
    filename = f"{model_name}_metrics.json"
    filepath = os.path.join(metrics_dir, filename)

    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return 0

    try:
        with open(filepath) as f:
            metrics = json.load(f)
        return metrics.get(metric_name, 0)
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return 0

def main():
    models = ["CatBoost", "LightGBM", "RandomForest", "XGBoost"]
    metric_name = "roc_auc"

    # Load metrics
    scores = {}
    for model in models:
        score = load_metric(model, metric_name)
        scores[model] = score
        print(f"{model}: {score:.4f}")

    # Debug: list actual files in artifacts
    eval_dir = os.path.join("artifacts", "model_evaluation")
    if os.path.exists(eval_dir):
        print("\n📁 Files in artifacts/model_evaluation/:")
        for f in os.listdir(eval_dir):
            print(f"  - {f}")
    else:
        print(f"\n❌ Directory not found: {eval_dir}")

    # Select best model
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    if best_score <= 0:
        raise ValueError(f"\n❌ No valid metrics found for promotion ({metric_name}) — aborting pipeline")
        sys.exit(1)

    print(f"\n🏆 Best model: {best_model} ({metric_name} = {best_score:.4f})")

    # Promote to Staging in MLflow 
    try:
        client = mlflow.MlflowClient()
        registered_name = f"{best_model}Model"
        latest = client.get_latest_versions(registered_name)[0]
        version = latest.version

        print(f"→ Promoting {registered_name} v{version} to 'Staging'...")

        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Staging",  # note: MLflow 2.2.2 uses capitalized stage names
            archive_existing_versions=True
        )

        print("✅ Promotion successful! Version is now in 'Staging'.")

    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
