import json
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pathlib import Path

class MLFlowUtils:

    def __init__(self, config_name = None):
        config_path = Path("./demo2/config/experiment_config.json")
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = json.load(file)[config_name]

        self.params = self.config.get("params", {})
        self.tags = self.config.get("tags", {})

        #self.experiment_name = self.config.get("experiment_name")
        #mlflow.set_experiment(self.experiment_name)

    def log_params(self):
        """Registra los parámetros en MLflow"""
        mlflow.log_params(self.params)

    def log_tags(self):
        """Registra los tags en MLflow"""
        mlflow.set_tags(self.tags)

    def log_model(self, model, artifact_path, signature=None):
        if signature:
            mlflow.sklearn.log_model(model, artifact_path, signature=signature)
        else:
            mlflow.sklearn.log_model(model, artifact_path)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_artifact(self, artifact_path):
        mlflow.log_artifact(artifact_path)

    def end_run(self):
        mlflow.end_run()

    def load_model(self, run_id, artifact_path):
        """Carga un modelo desde MLflow basado en run_id y artifact_path."""
        logged_model = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(logged_model)

    def register_model(self, run_id, artifact_path, model_name):
        """Registra el modelo en MLflow."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow.register_model(model_uri, model_name)

        # Devuelve la última versión del modelo registrado
        client = MlflowClient()
        model_version_details = client.get_latest_versions(model_name, stages=["None"])
        if model_version_details:
            return model_version_details[0].version
        return None

    def set_model_stage(self, model_name, version, stage):
        client = MlflowClient()
        client.transition_model_version_stage(name=model_name, version=version, stage=stage)

