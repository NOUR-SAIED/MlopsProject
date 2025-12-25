pipeline {
    agent any

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
                sh 'rm -rf artifacts/ || echo "No artifacts/ to remove"'
                sh 'find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'python3 -m pip install -r requirements.txt'
            }
        }

        stage('Run Training') {
            steps {
                sh '''
                    export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
                    python3 main.py
                '''
            }
        }
    }

    post {
        success {
            echo "✅ Training succeeded — check DagsHub!"
        }
        failure {
            echo "❌ Training failed — check logs"
        }
    }
}