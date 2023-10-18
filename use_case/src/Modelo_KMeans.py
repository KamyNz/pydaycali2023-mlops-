import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import mlflow
import json
from mlflow_utils import MLFlowUtils
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel

def cargar_datos():
    """Carga los datos desde un archivo CSV."""
    camino = Path('./demo2/data/analytics/pqrTipoPQRClasificado.csv')
    data = pd.read_csv(camino)
    return data

def procesar_datos(data):
    """Procesa los datos para el modelo KMeans."""
    data['FECHA_RADICADO'] = pd.to_datetime(data['FECHA_RADICADO'], errors='coerce', format='%Y-%m-%d')
    hoy = datetime.now()
    data['dias'] = (hoy - data['FECHA_RADICADO']).dt.days
    X = data[['Cantidad Bigramas', 'Cantidad Trigramas', 'Topico', 'Cantidad Verbos', 'Cantidad Adjetivos', 'Cantidad Adverbios', 'Cantidad Sustantivos', 'Cantidad_Palabras', 'Promedio_Palabras', 'DIVERSIDAD_LEXICA']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def entrenar_kmeans(X, n_clusters):
    """Entrena el modelo KMeans con un número determinado de clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return kmeans, labels

def asignar_criticidad(cluster):
    """Asigna una criticidad basada en el cluster."""
    if cluster == 0:
        return "alta"
    elif cluster == 1:
        return "media"
    elif cluster == 2:
        return "baja"
    else:
        return "muy baja"

def guardar_datos(data):
    """Guarda los datos con la criticidad en un archivo CSV."""
    data.to_csv(Path("../data/analytics/dataPredicciones.csv"), index=False)

def guardar_modelo_kmeans(modelo, path_modelo):
    """Guarda el modelo en un archivo .pkl."""
    joblib.dump(modelo, path_modelo)

if __name__ == "__main__":
    url_uri = "http://localhost:5000"
    experiment_name = "puj-202301-poc-02-sde-transporte-demo"

    # Setting tracking uri in localhost
    mlflow.set_tracking_uri(url_uri)

    print(f'url_uri used is: {url_uri}')
    print(f'The experiment to configure is: {experiment_name}')

    # Setting experiment name in localhost URI
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))

#################################################################################
    #Using MLFLowUtils module to set tags and params
    # tracker = MLFlowUtils(config_name="Modelo_KMeans")

    # # Cargar y procesar datos
    # data = cargar_datos()
    # X_scaled = procesar_datos(data)

    #//TODO: Function 1 here

    # clusters = 5
    # for k in range(2, clusters):
    #     with mlflow.start_run(experiment_id=experiment.experiment_id):
    #         # Tracking Parameters
    #         mlflow.log_param("n_clusters", k)

    #         kmeans = KMeans(n_clusters=k, random_state=42)
    #         kmeans.fit(X_scaled)
    #         labels = kmeans.labels_

    #         # Calculate and log metrics
    #         inertia = kmeans.inertia_
    #         silhouette_avg = silhouette_score(X_scaled, labels)
    #         dunn_score = davies_bouldin_score(X_scaled, labels)

    #         # Tracking Metrics
    #         mlflow.log_metric("Inertia", inertia)
    #         mlflow.log_metric("Silhouette_Score", silhouette_avg)
    #         mlflow.log_metric("Dunn_Index", dunn_score)

    #         tracker.log_tags()

    #         #Log model using infer_signature
    #         signature = infer_signature(pd.DataFrame(X_scaled), kmeans.predict(X_scaled))
    #         mlflow.sklearn.log_model(kmeans, "kmeans_model_"+str(k), signature=signature)

    #//TODO: Function 2 here
    # # # Registrar el modelo
    # model_name = "kmeans_model_4"
    # run_id = "5d2f9ca3a22e45429738b4786e1f86ad"
    # registered_model_name="kmeans_model_4_V1"

    # # Get the run object using the run_id
    # run_info = mlflow.get_run(run_id)

    # # Retrieve the artifact path for the run
    # artifact_uri = run_info.info.artifact_uri
    # model_uri=f"runs:/{run_id}/{model_name}"

    # print(run_info)
    # print(artifact_uri)
    # print(model_uri)

    # # Register the model using the artifact path
    # registered_model = mlflow.register_model(
    # model_uri=f"runs:/{run_id}/{model_name}",
    # name=registered_model_name,
    # await_registration_for=300  # Optional: Wait for up to 5 minutes (300 seconds) for registration completion
    # )

    # print(f"Model registered: {registered_model}")

    # //TODO: Function 3 here

    # registered_model_name="kmeans_model_4_V1"
    # client = MlflowClient()
    # model_details = client.get_registered_model(registered_model_name)
    # latest_version = max([int(v.version) for v in model_details.latest_versions])
    # print(latest_version)

    # # Transicionar la última versión del modelo a "Staging"
    # client = MlflowClient()
    # client.transition_model_version_stage(
    #     name=registered_model_name,
    #     version=latest_version,
    #     stage="Staging"
    # )

    #//TODO: Function 4 here

    # # Usar el run_id para definir logged_model
    # logged_model = f"runs:/{run_id}/{model_name}"

    # # Cargar el modelo como un PyFuncModel
    # loaded_model = mlflow.sklearn.load_model(logged_model)

    # # Hacer predicciones en un DataFrame de pandas
    # predictions = loaded_model.predict(pd.DataFrame(X_scaled))

    # # Añadir la criticidad al dataset
    # data['cluster'] = loaded_model.labels_
    # data['Criticidad'] = data['cluster'].apply(asignar_criticidad)
    # data['Predicciones'] = predictions

    # print(data.head())

    # with mlflow.start_run(run_id=run_id):

    #     # Log the artifact to the current run
    #     local_artifact_path = './demo2/data/analytics/data_with_predictions_conModeloCargado.csv'  # Replace this with the path to your local artifact
    #     mlflow.log_artifact(local_artifact_path)

    #     # Guardar los resultados y hacer seguimiento del archivo
    #     data.to_csv(Path(local_artifact_path),
    #                 index=False)
    #     tracker.log_artifact(local_artifact_path)

    # Finalizar el seguimiento con MLflow
    #tracker.end_run()
#################################################################################



