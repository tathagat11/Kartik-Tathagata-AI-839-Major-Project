import subprocess
import mlflow
import os
import sys
import pytest

from kedro.framework.hooks import hook_impl

class TestingHook:
    @hook_impl
    def before_pipeline_run(self):
        print("Running tests before pipeline execution...")
        # Run pytest and capture the result
        result = pytest.main(['tests/', '-v'])
        
        if result != pytest.ExitCode.OK:
            print("Tests failed! Pipeline execution stopped.")
            sys.exit(1)
        
        print("All tests passed. Continuing with pipeline execution...")

class MLflowModelDeploymentHook:
    @hook_impl
    def after_pipeline_run(self):
        """
            Runs after all pipelines finish running and deploys the latest model to a Docker container
            Args:
                None
            Returns:
                None
        """
        try:
            # Get latest model version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions("review_rating_model", stages=["None"])[0]
            latest_version = client.get_model_version("review_rating_model", 9)

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