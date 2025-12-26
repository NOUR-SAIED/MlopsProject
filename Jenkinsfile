pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = "https://dagshub.com/hannamhiri/MlopsProject.mlflow"
        MLFLOW_TRACKING_USERNAME = "hannamhiri"
        MLFLOW_TRACKING_PASSWORD = credentials('DAGSHUB_TOKEN')
    }

    stages {

        stage('SCM Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Cleanup') {
            steps {
                sh 'rm -rf artifacts venv || true'
            }
        }

        stage('Train & Register Model (Python 3.11)') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    args '-u root'
                }
            }
            steps {
                sh '''
                    apt-get update
                    apt-get install -y libgomp1 gcc
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt

                    python main.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build -t noursaied622/mlops-app:latest .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'DOCKER_HUB') {
                        docker.image('noursaied622/mlops-app:latest').push('latest')
                    }
                }
            }
        }
    }
}
