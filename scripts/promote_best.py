import json
import mlflow

def load_metric(model_name, metric_name):
    """Load metric from local JSON (faster than querying MLflow)"""
    try:
        with open(f"artifacts/model_evaluation/{model_name}_metrics.json") as f:
            metrics = json.load(f)
        return metrics.get(metric_name, 0)
    except Exception as e:
        print(f"⚠️ Could not load {model_name} metrics: {e}")
        return 0

def main():
    # Define models to compare
    models = ["CatBoost", "LightGBM", "RandomForest", "XGBoost"]
    metric = "f1_score"  # or "accuracy", "auc", etc.

    # Load scores
    scores = {}
    for model in models:
        score = load_metric(model, metric)
        scores[model] = score
        print(f"{model}: {score:.4f}")

    # Pick best
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    if best_score == 0:
        print("❌ No valid metrics found — skipping promotion")
        return

    print(f"\n🏆 Best model: {best_model} ({metric} = {best_score:.4f})")

    # Promote to Staging
    try:
        client = mlflow.MlflowClient()
        registered_name = f"{best_model}Model"
        latest = client.get_latest_versions(registered_name)[0]
        version = latest.version

        print(f"→ Promoting {registered_name} v{version} to 'Staging'...")
        client.transition_model_version_stage(
            name=registered_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True
        )
        print("✅ Promotion successful!")

    except Exception as e:
        print(f"❌ Promotion failed: {e}")

if __name__ == "__main__":
    main()