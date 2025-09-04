import joblib

def save_pipeline(pipeline, path="models/titanic_logreg_pipeline.joblib"):
    """Save the trained pipeline to disk."""
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")

def load_pipeline(path="models/titanic_logreg_pipeline.joblib"):
    """Load the trained pipeline from disk."""
    pipeline = joblib.load(path)
    print(f"Model loaded from {path}")
    return pipeline
