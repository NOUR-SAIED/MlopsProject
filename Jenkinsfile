pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.io' // Change this to your Docker registry
        IMAGE_NAME = 'mlops-project'
        KUBERNETES_NAMESPACE = 'mlops'
        DOCKER_IMAGE_TRAIN = "${DOCKER_REGISTRY}/${IMAGE_NAME}-train:${BUILD_NUMBER}"
        DOCKER_IMAGE_SERVE = "${DOCKER_REGISTRY}/${IMAGE_NAME}-serve:${BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig') // Jenkins credential ID for kubeconfig
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Build Training Image') {
            steps {
                script {
                    echo "Building training Docker image..."
                    sh """
                        docker build -f Dockerfile.train -t ${DOCKER_IMAGE_TRAIN} .
                        docker tag ${DOCKER_IMAGE_TRAIN} ${DOCKER_REGISTRY}/${IMAGE_NAME}-train:latest
                    """
                }
            }
        }
        
        stage('Build Serving Image') {
            steps {
                script {
                    echo "Building serving Docker image..."
                    sh """
                        docker build -f Dockerfile.serve -t ${DOCKER_IMAGE_SERVE} .
                        docker tag ${DOCKER_IMAGE_SERVE} ${DOCKER_REGISTRY}/${IMAGE_NAME}-serve:latest
                    """
                }
            }
        }
        
        stage('Push Images') {
            steps {
                script {
                    echo "Pushing Docker images to registry..."
                    withCredentials([usernamePassword(credentialsId: 'docker-registry-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh """
                            echo ${DOCKER_PASS} | docker login ${DOCKER_REGISTRY} -u ${DOCKER_USER} --password-stdin
                            docker push ${DOCKER_IMAGE_TRAIN}
                            docker push ${DOCKER_IMAGE_SERVE}
                            docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}-train:latest
                            docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}-serve:latest
                        """
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    echo "Deploying to Kubernetes..."
                    sh """
                        kubectl --kubeconfig=${KUBECONFIG} create namespace ${KUBERNETES_NAMESPACE} --dry-run=client -o yaml | kubectl --kubeconfig=${KUBECONFIG} apply -f -
                        kubectl --kubeconfig=${KUBECONFIG} apply -f k8s/ -n ${KUBERNETES_NAMESPACE}
                        kubectl --kubeconfig=${KUBECONFIG} set image deployment/mlops-app mlops-app=${DOCKER_IMAGE_SERVE} -n ${KUBERNETES_NAMESPACE}
                        kubectl --kubeconfig=${KUBECONFIG} rollout status deployment/mlops-app -n ${KUBERNETES_NAMESPACE}
                    """
                }
            }
        }
        
        stage('Health Check') {
            steps {
                script {
                    echo "Performing health check..."
                    sh """
                        sleep 10
                        kubectl --kubeconfig=${KUBECONFIG} get pods -n ${KUBERNETES_NAMESPACE} -l app=mlops-app
                        kubectl --kubeconfig=${KUBECONFIG} get svc -n ${KUBERNETES_NAMESPACE} mlops-service
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo "Pipeline succeeded! Application deployed successfully."
            // Optional: Send notification
        }
        failure {
            echo "Pipeline failed! Check logs for details."
            // Optional: Send notification
        }
        always {
            cleanWs()
        }
    }
}

