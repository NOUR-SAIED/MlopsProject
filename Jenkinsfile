pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "https://dagshub.com/hannamhiri/MlopsProject.mlflow"
        MLFLOW_TRACKING_USERNAME = "hannamhiri"
        MLFLOW_TRACKING_PASSWORD = credentials('DAGSHUB_TOKEN')
        IMAGE_NAME = "noursaied622/mlops-app"
    }

    stages {

        stage('SCM Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Cleanup') {
            steps {
                sh '''
                    rm -rf artifacts venv
                '''
            }
        }

        stage('Train & Register Model (Python 3.11)') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    args '-u root'
                    reuseNode true
                }
            }
            steps {
                sh '''
                    set -e

                    apt-get update
                    apt-get install -y libgomp1 gcc

                    python --version

                    pip install --upgrade pip
                    pip install -r requirements.txt

                    python main.py

                    echo "Artifacts generated:"
                    ls -R artifacts
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    ls -R artifacts
                    docker build -t $IMAGE_NAME:latest .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'DOCKER_HUB') {
                        docker.image("$IMAGE_NAME:latest").push('latest')
                    }
                }
            }
        }
    }
}
