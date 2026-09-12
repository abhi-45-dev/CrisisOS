"""Train the Flood v2 supervised ML model."""

from ml.flood_ml import train_flood_model


if __name__ == "__main__":
    print("=== FLOOD V2 TRAINING ===")
    print("Downloading/locating the real India Flood Inventory if needed...")
    result = train_flood_model()
    print("\n=== FLOOD V2 TRAINING COMPLETE ===")
    print(f"Model: {result['algorithm']}")
    print(f"Dataset: {result['dataset_file']}")
    print(f"Rows: {result['rows']}")
    print(f"Training years: {result['training_years']}")
    print(f"Validation years: {result['validation_years']}")
    print(f"Test years: {result['test_years']}")
    test = result["metrics"]["test"]
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test macro F1: {test['f1_macro']:.4f}")
    print(f"Test ROC-AUC: {test.get('roc_auc')}")
    print("Top permutation features:")
    for row in result["feature_importance"][:10]:
        print(f"  {row['feature']}: {row['importance_mean']:.6f}")
    print("\nArtifacts written under backend/ml/models/flood/v1/")
