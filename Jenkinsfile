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
                sh 'rm -rf artifacts/ venv/ || echo "Cleanup"'
                sh 'find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
            }
        }

        stage('Install Python 3.11') {
            steps {
                sh '''
                    # Install Python 3.11 (supports scikit-learn 1.2.2)
                    apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev
                '''
            }
        }

        stage('Setup Python 3.11 Environment') {
            steps {
                sh '''
                    # Create venv with Python 3.11
                    python3.11 -m venv venv
                    . venv/bin/activate

                    # Upgrade pip & install EXACT requirements (no changes!)
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Training') {
            steps {
                sh '''
                    # Activate venv and run
                    . venv/bin/activate
                    export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
                    python main.py
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