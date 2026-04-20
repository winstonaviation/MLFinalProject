import numpy as np
import time
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)

def evaluate_model(model, X_test, y_test, model_name, binary=True):
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    results = {
        "model": model_name,
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred),
        "inference_time_sec": inference_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    #aUROC: binary uses probability of positive class; multiclass uses OvR weighted
    if binary:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred
        results["auroc"] = roc_auc_score(y_test, y_prob)
    else:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            results["auroc"] = roc_auc_score(
                y_test, y_prob, multi_class='ovr', average='weighted'
            )

    return results


def print_results(results):
    print(f"\n=== {results['model']} ===")
    print(f"F1 Score:       {results['f1']:.4f}")
    print(f"Accuracy:       {results['accuracy']:.4f}")
    if "auroc" in results:
        print(f"AUROC:          {results['auroc']:.4f}")
    print(f"Inference Time: {results['inference_time_sec']:.4f}s")
    print(f"\nClassification Report:\n{results['classification_report']}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")