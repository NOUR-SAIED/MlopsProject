import json
import os
import mlflow

def load_metric(model_name, metric_name):
    """Load metric from local JSON (OS-agnostic path)"""
    # Use os.path.join for cross-platform compatibility
    metrics_dir = os.path.join("artifacts", "model_evaluation")
    filename = f"{model_name}_metrics.json"
    filepath = os.path.join(metrics_dir, filename)
    
    try:
        if os.path.exists(filepath):
            with open(filepath) as f:
                metrics = json.load(f)
            return metrics.get(metric_name, 0)
        else:
            print(f"⚠️ File not found: {filepath}")
            return 0
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return 0

def main():
    models = ["CatBoost", "LightGBM", "RandomForest", "XGBoost"]
    metric = "f1_score"

    scores = {}
    for model in models:
        score = load_metric(model, metric)
        scores[model] = score
        print(f"{model}: {score:.4f}")

    # Debug: list actual files in artifacts/model_evaluation/
    eval_dir = os.path.join("artifacts", "model_evaluation")
    if os.path.exists(eval_dir):
        print("\n📁 Files in artifacts/model_evaluation/:")
        for f in os.listdir(eval_dir):
            print(f"  - {f}")
    else:
        print(f"\n❌ Directory not found: {eval_dir}")

    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    if best_score <= 0:
        print("\n❌ No valid metrics found — skipping promotion")
        return

    print(f"\n🏆 Best model: {best_model} ({metric} = {best_score:.4f})")

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