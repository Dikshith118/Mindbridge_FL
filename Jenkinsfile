// ══════════════════════════════════════════════════════════════════════════
// MindBridge — Jenkins Pipeline
// Requires these Jenkins plugins:
//   - Pipeline, Git
//   - SonarQube Scanner for Jenkins
//   - OWASP Dependency-Check Plugin
//   - Docker Pipeline
//   - SSH Agent
//   - Credentials Binding
//
// Requires these Jenkins credentials configured beforehand (Manage Jenkins
// → Credentials):
//   sonarqube-token        (Secret text)  — SonarQube auth token
//   ghcr-credentials        (Username/Password) — GHCR (or your registry) login
//   vm-ssh-key               (SSH Username with private key) — deploy target
//   mindbridge-domain       (Secret text)  — e.g. mindbridge.yourdomain.com
//
// Requires a SonarQube server configured under:
//   Manage Jenkins → System → SonarQube servers → name it "sonarqube"
// ══════════════════════════════════════════════════════════════════════════

pipeline {
    agent any

    options {
        timeout(time: 45, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    environment {
        REGISTRY       = 'ghcr.io'
        REPO_OWNER     = 'your-org'   // ← replace with real GitHub org/user
        SERVER_IMAGE   = "${REGISTRY}/${REPO_OWNER}/mindbridge-server"
        CLIENT_IMAGE   = "${REGISTRY}/${REPO_OWNER}/mindbridge-client"
        GIT_SHA        = "${GIT_COMMIT.take(8)}"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python env') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install flake8 pylint
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . .venv/bin/activate
                    # Fail the build only on real errors (syntax/undefined names)
                    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
                    # Style issues reported but non-blocking
                    flake8 . --count --exit-zero --max-line-length=120 --statistics > flake8-report.txt
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'flake8-report.txt', allowEmptyArchive: true
                }
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pip install pytest pytest-flask pytest-cov
                    MINDBRIDGE_TEST_MODE=1 pytest tests/ -v --tb=short \
                        --junitxml=test-results.xml \
                        --cov=. --cov-report=xml:coverage.xml --cov-report=term
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withSonarQubeEnv('sonarqube') {
                    sh '''
                        sonar-scanner \
                          -Dsonar.projectKey=mindbridge \
                          -Dsonar.sources=. \
                          -Dsonar.exclusions=".venv/**,tests/**,client_data/**,saved_model/**,data/**" \
                          -Dsonar.python.coverage.reportPaths=coverage.xml \
                          -Dsonar.python.flake8.reportPaths=flake8-report.txt
                    '''
                }
            }
        }

        stage('Quality Gate') {
            steps {
                // Fails the build if SonarQube's configured quality gate fails.
                // webhook must be set on the SonarQube server pointing back to Jenkins.
                timeout(time: 10, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Dependency Check (OWASP)') {
            steps {
                dependencyCheck additionalArguments: '''
                    --scan .
                    --format ALL
                    --project mindbridge
                    --failOnCVSS 8
                    --disableAssembly
                    --suppression dependency-check-suppressions.xml
                ''', odcInstallation: 'owasp-dependency-check'

                dependencyCheckPublisher pattern: 'dependency-check-report.xml'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'dependency-check-report.*', allowEmptyArchive: true
                }
            }
        }

        stage('Build Docker Images') {
            steps {
                sh '''
                    docker build -f Dockerfile.server -t ${SERVER_IMAGE}:${GIT_SHA} -t ${SERVER_IMAGE}:latest .
                    docker build -f Dockerfile.client -t ${CLIENT_IMAGE}:${GIT_SHA} -t ${CLIENT_IMAGE}:latest .
                '''
            }
        }

        stage('Push Images') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-credentials',
                    usernameVariable: 'REG_USER',
                    passwordVariable: 'REG_PASS'
                )]) {
                    sh '''
                        echo "$REG_PASS" | docker login ${REGISTRY} -u "$REG_USER" --password-stdin
                        docker push ${SERVER_IMAGE}:${GIT_SHA}
                        docker push ${SERVER_IMAGE}:latest
                        docker push ${CLIENT_IMAGE}:${GIT_SHA}
                        docker push ${CLIENT_IMAGE}:latest
                    '''
                }
            }
        }

        stage('Deploy to VM') {
            when { branch 'main' }
            steps {
                withCredentials([
                    sshUserPrivateKey(credentialsId: 'vm-ssh-key', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER'),
                    string(credentialsId: 'mindbridge-domain', variable: 'MB_DOMAIN'),
                    string(credentialsId: 'vm-host', variable: 'VM_HOST')
                ]) {
                    sh '''
                        scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
                            docker-compose.yml proxy/Caddyfile \
                            "$SSH_USER@$VM_HOST:/opt/mindbridge/"

                        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$VM_HOST" "
                            cd /opt/mindbridge &&
                            echo 'SERVER_IMAGE=${SERVER_IMAGE}:${GIT_SHA}' > .env &&
                            echo 'MINDBRIDGE_DOMAIN=$MB_DOMAIN' >> .env &&
                            docker compose pull mindbridge-server retrainer &&
                            docker compose up -d --remove-orphans &&
                            docker image prune -f
                        "
                    '''
                }
            }
        }

        stage('Post-Deploy Health Check') {
            when { branch 'main' }
            steps {
                withCredentials([string(credentialsId: 'mindbridge-domain', variable: 'MB_DOMAIN')]) {
                    sh '''
                        for i in $(seq 1 10); do
                            if curl -sf "https://$MB_DOMAIN/status"; then
                                echo "Deployment healthy"
                                exit 0
                            fi
                            echo "Waiting for server... ($i/10)"
                            sleep 6
                        done
                        echo "Health check failed after deploy"
                        exit 1
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f || true'
        }
        failure {
            echo 'Pipeline failed — check Sonar/Dependency-Check/test reports above.'
        }
    }
}
