# Authenticate (Subsequent Sessions)

Run this every time you need cloud access and are not yet authenticated. The SessionStart hook normally handles this automatically, but this flow serves as a fallback.

1. Read `.cloud-config.json` to determine the provider.
2. **Check credential age:** If `created_at` exists in `.cloud-config.json`, calculate how old the credentials are. If older than **180 days**, warn the user:
   ```
   Your cloud credentials were created <N> days ago. Consider rotating
   them for security. See the "Credential Rotation" workflow.
   ```
   This is a warning only — do not block authentication.
3. Ensure the provider's CLI is installed by running the installation script from the corresponding reference file. This is a safety net in case the SessionStart hook hasn't run yet.
4. Get the current user's email:
   ```bash
   USER_EMAIL=$(git config user.email)
   ```
5. Read the corresponding provider reference file in this skill's directory.
6. Resolve the encryption key.
7. Determine the credential file path based on config format, then decrypt with restrictive permissions and guaranteed cleanup:
   ```bash
   # Check if config uses providers[] (multi-provider) or provider (single)
   if jq -e '.providers' .cloud-config.json >/dev/null 2>&1; then
     PROVIDER=$(jq -e -r '.provider' .cloud-config.json 2>/dev/null || jq -r '.providers[0].provider' .cloud-config.json)
     ENC_FILE=".cloud-credentials.${PROVIDER}.${USER_EMAIL}.enc"
     # Fall back to single-provider naming if provider-prefixed file doesn't exist
     [ ! -f "$ENC_FILE" ] && ENC_FILE=".cloud-credentials.${USER_EMAIL}.enc"
   else
     ENC_FILE=".cloud-credentials.${USER_EMAIL}.enc"
   fi
   trap 'rm -f /tmp/credentials.json' EXIT
   (umask 077 && echo "$KEY" | openssl enc -d -aes-256-cbc -pbkdf2 \
     -pass stdin \
     -in "$ENC_FILE" -out /tmp/credentials.json)
   ```
8. Activate using the provider-specific commands from the reference file. If activation fails, warn the user rather than leaving credentials on disk — the `EXIT` trap ensures cleanup.
9. **Delete `/tmp/credentials.json` immediately after activation** (the `EXIT` trap handles this automatically, but explicit removal is still recommended).
10. **Verify credentials work** by running the smoke test command from the provider reference file (see "Verify (Smoke Test)" section). If the smoke test fails, inform the user that credentials may be expired or revoked and suggest re-running setup.
