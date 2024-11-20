import subprocess
import mlflow
import os
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession


# class SparkHooks:
#     @hook_impl
#     def after_context_created(self, context) -> None:
#         """Initialises a SparkSession using the config
#         defined in project's conf folder.
#         """

#         # Load the spark configuration in spark.yaml using the config loader
#         parameters = context.config_loader["spark"]
#         spark_conf = SparkConf().setAll(parameters.items())

#         # Initialise the spark session
#         spark_session_conf = (
#             SparkSession.builder.appName(context.project_path.name)
#             .enableHiveSupport()
#             .config(conf=spark_conf)
#         )
#         _spark_session = spark_session_conf.getOrCreate()
#         _spark_session.sparkContext.setLogLevel("WARN")

class MLflowModelDeploymentHook:
    @hook_impl
    def after_pipeline_run(self):
        try:
            # Get latest model version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions("review_rating_model", stages=["None"])[0]
            
            # Get model path
            artifact_uri = latest_version.source
            local_path = artifact_uri.replace("file://", "")
            absolute_model_path = os.path.abspath(local_path)
            print(absolute_model_path)
            # Get deployment directory for Dockerfile
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            deployment_dir = os.path.join(project_root, "deployment", "app")
            print(deployment_dir)
            
            # Build the Docker image
            print("Building Docker image...")
            subprocess.run(
                ["docker", "build", "-t", "review-rating-server", deployment_dir],
                check=True
            )
            
            # Stop and remove existing container
            print("Cleaning up existing container...")
            subprocess.run(["docker", "stop", "review-rating-server"], check=False)
            subprocess.run(["docker", "rm", "review-rating-server"], check=False)
            
            # Start new container
            print("Starting new container...")
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "review-rating-server",
                    "-p",
                    "5002:5002",
                    "-v",
                    f"{absolute_model_path}:/app/model",
                    "review-rating-server",
                ],
                check=True,
            )
            
            print(f"Deployed latest model version: {latest_version.version}")
            print("API is available at http://localhost:5002")
            print("\nYou can test it with:")
            print("""
    curl -X POST http://localhost:5002/predict \\
         -H "Content-Type: application/json" \\
         -d '{"text": "Great product, highly recommend!"}'
    # """)
            
        except Exception as e:
            print(f"Error during model deployment: {str(e)}")
            raise