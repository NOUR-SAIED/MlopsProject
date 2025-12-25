pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-u root'
        }
    }

    environment {
        MLFLOW_TRACKING_URI = "https://dagshub.com/hannamhiri/MlopsProject.mlflow"
    }

    stages {
        stage('SCM Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Cleanup') {
            steps {
                sh 'rm -rf artifacts/ venv/ || true'
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate

                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Training') {
            steps {
                sh '''
                    . venv/bin/activate
                    export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
                    python main.py
                '''
            }
        }
    }
}
