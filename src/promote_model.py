from mlflow.tracking import MlflowClient

MODEL_NAME = "cats-dogs-resnet18"
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

best, best_acc = None, -1
for v in versions:
    acc = client.get_run(v.run_id).data.metrics.get("val_accuracy", 0)
    if acc > best_acc:
        best, best_acc = v, acc

if best:
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(MODEL_NAME, v.version, "Archived")

    client.transition_model_version_stage(MODEL_NAME, best.version, "Production")
    print(f"Promoted version {best.version} as Champion")
