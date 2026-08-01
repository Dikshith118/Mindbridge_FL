# Jenkins Setup — MindBridge Pipeline

This assumes Jenkins is already running. These are the one-time setup steps
before the `Jenkinsfile` will work.

## 1. Install required plugins
Manage Jenkins → Plugins → Available:

- **Pipeline** (usually pre-installed)
- **Git**
- **SonarQube Scanner for Jenkins**
- **OWASP Dependency-Check Plugin**
- **Docker Pipeline**
- **SSH Agent**
- **Credentials Binding**

## 2. Configure global tools
Manage Jenkins → Tools:

- **SonarQube Scanner** → add an installation (auto-install latest, or point
  to a manual install)
- **OWASP Dependency-Check** → add an installation named
  `owasp-dependency-check` (must match the name used in the Jenkinsfile's
  `odcInstallation` field) — this triggers an auto-download of the CVE
  database on first run

## 3. Connect Jenkins to your SonarQube server
Manage Jenkins → System → SonarQube servers:
- Name: `sonarqube` (must match `withSonarQubeEnv('sonarqube')` in the
  Jenkinsfile)
- Server URL: your SonarQube instance URL
- Token: create under SonarQube → My Account → Security → Generate Token,
  then store it as a Jenkins credential (see step 4) and select it here

Also set up the **webhook** on the SonarQube project (Administration →
Webhooks) pointing to `http://<jenkins-url>/sonarqube-webhook/` — without
this, the `waitForQualityGate` stage will hang until timeout instead of
getting an immediate pass/fail.

## 4. Add credentials
Manage Jenkins → Credentials → System → Global credentials → Add:

| ID                    | Type                          | Value                                    |
|------------------------|--------------------------------|--------------------------------------------|
| `sonarqube-token`      | Secret text                    | Token generated in step 3                 |
| `ghcr-credentials`     | Username with password         | Your container registry username + PAT/password |
| `vm-ssh-key`           | SSH Username with private key  | SSH user + private key for the deploy VM  |
| `vm-host`              | Secret text                    | VM's IP or hostname                       |
| `mindbridge-domain`    | Secret text                    | e.g. `mindbridge.yourdomain.com`          |

## 5. Create the pipeline job
New Item → Pipeline (or Multibranch Pipeline if you want PR/branch builds
automatically):
- Pipeline script from SCM → Git → your repo URL
- Script Path: `Jenkinsfile`

## 6. First run
Trigger a manual build. Expected stage order:

```
Checkout → Set up Python env → Lint → Unit Tests →
SonarQube Analysis → Quality Gate → Dependency Check (OWASP) →
Build Docker Images → Push Images* → Deploy to VM* → Health Check*

  * only on the `main` branch
```

If **Quality Gate** or **Dependency Check** fails the build, that's the
pipeline doing its job — fix the flagged issues rather than suppressing
them, except for confirmed false positives (see
`dependency-check-suppressions.xml`).

## 7. Ongoing
Every commit to `main` now: lints → tests → scans code quality → scans
dependencies for known CVEs (CVSS ≥ 8 fails the build) → builds and pushes
Docker images → deploys to the VM → verifies it's actually healthy after
deploy. Any stage failing stops the pipeline before it reaches deploy.
