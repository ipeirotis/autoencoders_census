# Multi-Provider Setup

A repo may need access to multiple cloud providers (e.g., GCP for BigQuery and AWS for S3). This skill supports this with a few conventions.

## Config Format

When a second provider is added, convert `.cloud-config.json` from a single-provider object to a `providers` array:

```json
{
  "providers": [
    {
      "provider": "gcp",
      "project_id": "my-gcp-project",
      "service_account": "claude-agent@my-gcp-project.iam.gserviceaccount.com",
      "roles": ["roles/storage.objectAdmin"],
      "created_at": "2025-03-15T10:00:00Z"
    },
    {
      "provider": "aws",
      "project_id": "123456789012",
      "service_account": "claude-agents",
      "roles": ["AmazonS3FullAccess"],
      "created_at": "2025-03-16T14:00:00Z"
    }
  ]
}
```

## Credential File Naming

With multiple providers, include the provider in the filename:

```
.cloud-credentials.<provider>.<email>.enc
```

For example: `.cloud-credentials.gcp.alice@example.com.enc` and `.cloud-credentials.aws.alice@example.com.enc`.

## Backward Compatibility

If `.cloud-config.json` has a top-level `provider` field (single-provider format), treat it as-is — no migration needed until a second provider is added. When adding a second provider:

1. Read the existing single-provider config.
2. Rewrite `.cloud-config.json` to the `providers` array format.
3. Rename existing `.cloud-credentials.<email>.enc` files to `.cloud-credentials.<provider>.<email>.enc`.
4. Update `.claude/hooks/cloud-auth.sh` to iterate over all providers.
5. Commit all changes.

## Authentication

When the config uses the `providers` array format, authenticate **all** providers during the Authenticate flow (or in the SessionStart hook). Each provider uses its own credentials key env var, falling back to `CLOUD_CREDENTIALS_KEY`.

## Multi-Provider SessionStart Hook

When converting to multi-provider, replace the single-provider `cloud-auth.sh` with a script that iterates over all providers. The hook should:

1. Read the `providers` array from `.cloud-config.json`
2. For each provider entry, resolve the provider-specific credentials key
3. Look for the provider-prefixed credential file: `.cloud-credentials.<provider>.<email>.enc`
4. Decrypt and activate using the provider-specific commands from each reference file
5. Install each provider's CLI if missing

```bash
#!/bin/bash
set -e

CONFIG=".cloud-config.json"
if [ ! -f "$CONFIG" ]; then exit 0; fi

USER_EMAIL=$(git config user.email 2>/dev/null || true)
if [ -z "$USER_EMAIL" ]; then exit 0; fi

PROVIDER_COUNT=$(jq -r '.providers | length' "$CONFIG" 2>/dev/null)
if [ -z "$PROVIDER_COUNT" ] || [ "$PROVIDER_COUNT" = "null" ]; then exit 0; fi

for i in $(seq 0 $((PROVIDER_COUNT - 1))); do
  PROVIDER=$(jq -r ".providers[$i].provider" "$CONFIG")
  ENC_FILE=".cloud-credentials.${PROVIDER}.${USER_EMAIL}.enc"
  if [ ! -f "$ENC_FILE" ]; then continue; fi

  case "$PROVIDER" in
    gcp)   KEY="${GCP_CREDENTIALS_KEY:-$CLOUD_CREDENTIALS_KEY}" ;;
    aws)   KEY="${AWS_CREDENTIALS_KEY:-$CLOUD_CREDENTIALS_KEY}" ;;
    azure) KEY="${AZURE_CREDENTIALS_KEY:-$CLOUD_CREDENTIALS_KEY}" ;;
    *)     KEY="$CLOUD_CREDENTIALS_KEY" ;;
  esac
  if [ -z "$KEY" ]; then continue; fi

  (umask 077 && echo "$KEY" | openssl enc -d -aes-256-cbc -pbkdf2 \
    -pass stdin -in "$ENC_FILE" -out /tmp/credentials.json 2>/dev/null) || continue

  # Activate using provider-specific commands (install CLI + authenticate)
  # Each provider block is guarded so one failure doesn't block the others
  # See each provider's reference file for the full activation commands
  case "$PROVIDER" in
    gcp)
      if ! command -v gcloud &>/dev/null; then
        if ! curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/home/user; then
          echo "WARNING: gcloud SDK install failed — skipping GCP auth."
          rm -f /tmp/credentials.json; continue
        fi
        export PATH="/home/user/google-cloud-sdk/bin:$PATH"
      fi
      if ! gcloud auth activate-service-account --key-file=/tmp/credentials.json 2>/dev/null; then
        echo "WARNING: GCP auth failed — credentials may be revoked."
        rm -f /tmp/credentials.json; continue
      fi
      PROJECT_ID="$(jq -r ".providers[$i].project_id" "$CONFIG")"
      if ! gcloud config set project "$PROJECT_ID" 2>/dev/null; then
        echo "WARNING: Failed to set GCP project to $PROJECT_ID."
      fi
      ;;
    aws)
      if ! command -v aws &>/dev/null; then
        if ! (curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
          && unzip -q /tmp/awscliv2.zip -d /tmp \
          && /tmp/aws/install --install-dir /home/user/aws-cli --bin-dir /home/user/bin); then
          echo "WARNING: AWS CLI install failed — skipping AWS auth."
          rm -rf /tmp/awscliv2.zip /tmp/aws /tmp/credentials.json; continue
        fi
        export PATH="/home/user/bin:$PATH"
        rm -rf /tmp/awscliv2.zip /tmp/aws
      fi
      export AWS_ACCESS_KEY_ID=$(jq -r .access_key_id /tmp/credentials.json)
      export AWS_SECRET_ACCESS_KEY=$(jq -r .secret_access_key /tmp/credentials.json)
      export AWS_DEFAULT_REGION=$(jq -r .region /tmp/credentials.json)
      if [ -n "$CLAUDE_ENV_FILE" ]; then
        echo "export AWS_ACCESS_KEY_ID='$AWS_ACCESS_KEY_ID'" >> "$CLAUDE_ENV_FILE"
        echo "export AWS_SECRET_ACCESS_KEY='$AWS_SECRET_ACCESS_KEY'" >> "$CLAUDE_ENV_FILE"
        echo "export AWS_DEFAULT_REGION='$AWS_DEFAULT_REGION'" >> "$CLAUDE_ENV_FILE"
      fi
      ;;
    azure)
      if ! command -v az &>/dev/null; then
        if ! curl -sSL https://aka.ms/InstallAzureCLIDeb | sudo bash; then
          echo "WARNING: Azure CLI install failed — skipping Azure auth."
          rm -f /tmp/credentials.json; continue
        fi
      fi
      if ! az login --service-principal \
        --username "$(jq -r .appId /tmp/credentials.json)" \
        --password "$(jq -r .password /tmp/credentials.json)" \
        --tenant "$(jq -r .tenant /tmp/credentials.json)" 2>/dev/null; then
        echo "WARNING: Azure auth failed — credentials may be revoked."
        rm -f /tmp/credentials.json; continue
      fi
      SUBSCRIPTION="$(jq -r ".providers[$i].project_id" "$CONFIG")"
      if ! az account set --subscription "$SUBSCRIPTION" 2>/dev/null; then
        echo "WARNING: Failed to set Azure subscription to $SUBSCRIPTION."
      fi
      ;;
  esac

  rm -f /tmp/credentials.json
  echo "$PROVIDER credentials activated for $USER_EMAIL"
done
```
