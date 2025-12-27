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
                sh 'docker build -f Dockerfile.train -t mlops-train:latest .'
                sh 'docker run --rm -e MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} mlops-train:latest'
            }
        }

        stage('Promote Best Model') {
            steps {
                sh 'python promote_best.py'
            }
        }

        stage('Build Serving Docker Image') {
            steps {
                sh 'docker build -f Dockerfile.serve -t noursaied622/mlops-app:latest .'
            }
        }

        stage('Push Serving Image') {
            steps {
                script {
                    docker.withRegistry('', DOCKER_HUB_CREDENTIALS) {
                        docker.image('noursaied622/mlops-app:latest').push('latest')
                    }
                }
            }
        }
    }
}
