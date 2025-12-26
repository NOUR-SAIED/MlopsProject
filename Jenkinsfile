pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-u root'
        }
    }

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
                sh 'rm -rf artifacts/ venv/ || true'
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                    apt-get update
                    apt-get install -y libgomp1
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

        stage('Promote Best Model') {
            steps {
                sh '''
                    . venv/bin/activate
                    python scripts/promote_best.py
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                 . venv/bin/activate
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