#!/bin/bash
set -e

# --- Check credential prerequisites before doing anything expensive ---
CONFIG=".cloud-config.json"
if [ ! -f "$CONFIG" ]; then exit 0; fi

PROVIDER=$(jq -r .provider "$CONFIG" 2>/dev/null)
if [ "$PROVIDER" != "gcp" ]; then exit 0; fi

USER_EMAIL=$(git config user.email 2>/dev/null || true)
ENC_FILE=".cloud-credentials.${USER_EMAIL}.enc"
if [ -z "$USER_EMAIL" ] || [ ! -f "$ENC_FILE" ]; then exit 0; fi

KEY="${GCP_CREDENTIALS_KEY:-$CLOUD_CREDENTIALS_KEY}"
if [ -z "$KEY" ]; then exit 0; fi

# --- Install gcloud if missing (only after confirming auth is possible) ---
if ! command -v gcloud &> /dev/null; then
  if ! curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/home/user; then
    echo "WARNING: gcloud SDK install failed — skipping GCP auth."
    exit 0
  fi
  export PATH="/home/user/google-cloud-sdk/bin:$PATH"
fi

# --- Decrypt and activate credentials ---
(umask 077 && echo "$KEY" | openssl enc -d -aes-256-cbc -pbkdf2 \
  -pass stdin -in "$ENC_FILE" -out /tmp/credentials.json 2>/dev/null) || exit 0

trap 'rm -f /tmp/credentials.json' EXIT

if ! gcloud auth activate-service-account --key-file=/tmp/credentials.json 2>/dev/null; then
  echo "WARNING: GCP auth failed — credentials may be revoked. Run credential rotation to fix."
  exit 0
fi
PROJECT_ID="$(jq -r .project_id "$CONFIG")"
if ! gcloud config set project "$PROJECT_ID" 2>/dev/null; then
  echo "WARNING: Failed to set GCP project to $PROJECT_ID — verify project_id in .cloud-config.json."
fi
rm -f /tmp/credentials.json

echo "GCP credentials activated for $USER_EMAIL"
