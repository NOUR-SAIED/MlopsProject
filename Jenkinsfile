pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "https://dagshub.com/hannamhiri/MlopsProject.mlflow"
        MLFLOW_TRACKING_USERNAME = "hannamhiri"
        MLFLOW_TRACKING_PASSWORD = credentials('DAGSHUB_TOKEN')
        IMAGE_NAME = "noursaied622/mlops-app"
        DOCKER_HUB_CREDENTIALS = 'DOCKER_HUB'
    }

    stages {

        stage('SCM Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Train Model') {
            steps {
                echo "Building and running training container..."
                sh 'docker build -f Dockerfile.train -t mlops-train:latest .'
                sh 'docker run --rm -e MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} mlops-train:latest'
            }
        }

        stage('Promote Best Model') {
            steps {
                echo "Promoting the best model using training container..."
                sh 'docker run --rm -e MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} mlops-train:latest python promote_best.py'
            }
        }

        stage('Build Serving Docker Image') {
            steps {
                echo "Building the serving image..."
                sh 'docker build -f Dockerfile.serve -t ${IMAGE_NAME}:latest .'
            }
        }

        stage('Push Serving Image') {
            steps {
                echo "Pushing serving image to Docker Hub..."
                script {
                    docker.withRegistry('', DOCKER_HUB_CREDENTIALS) {
                        docker.image("${IMAGE_NAME}:latest").push('latest')
                    }
                }
            }
        }
    }
}
