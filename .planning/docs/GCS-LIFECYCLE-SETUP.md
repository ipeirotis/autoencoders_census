# GCS Lifecycle Management Setup

**Purpose:** Automatically delete uploaded CSVs and result files after 7-day retention period to prevent storage cost accumulation.

## Current Configuration

**Bucket:** `autoencoder_data` (from environment variable `GCS_BUCKET_NAME`)
**Retention period:** 7 days
**Applies to:** All files (uploads/, results/ prefixes)
**Firestore behavior:** Job metadata persists after GCS files deleted

## Lifecycle Rule Details

**Rule type:** Delete
**Condition:** Age > 7 days
**Action:** Permanent deletion (no soft delete)

Files are evaluated once per day. Changes to lifecycle rules take up to 24 hours to propagate.

## Setup Instructions

### Using gcloud CLI (Recommended)

1. **Create `lifecycle.json` configuration file:**

```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 7
        }
      }
    ]
  }
}
```

2. **Apply to bucket:**

```bash
gcloud storage buckets update gs://${GCS_BUCKET_NAME} --lifecycle-file=lifecycle.json
```

Replace `${GCS_BUCKET_NAME}` with your actual bucket name (e.g., `autoencoder_data`).

3. **Verify rule applied:**

```bash
gcloud storage buckets describe gs://${GCS_BUCKET_NAME} --format="json" | grep -A 10 "lifecycle"
```

Expected output:
```json
"lifecycle": {
  "rule": [
    {
      "action": {
        "type": "Delete"
      },
      "condition": {
        "age": 7
      }
    }
  ]
}
```

### Using Google Cloud Console

1. Navigate to [Cloud Storage Browser](https://console.cloud.google.com/storage/browser)
2. Click on bucket name (e.g., `autoencoder_data`)
3. Click "LIFECYCLE" tab
4. Click "ADD A RULE"
5. Select action: "Delete object"
6. Click "CONTINUE"
7. Set condition: "Age" = 7 days
8. Click "CONTINUE", then "CREATE"

**Note:** The console shows confirmation before creating the rule, unlike CLI which applies immediately.

## Expired Job Handling

When GCS files are auto-deleted after 7 days:

1. **Firestore:** Job document remains (status, createdAt, userId, metadata preserved)
2. **Frontend:** `/job/:id` page should show "Files expired - data deleted after 7 days" message
3. **Download button:** Should be hidden for expired jobs (age check: `createdAt + 7 days < now`)
4. **API behavior:** GET `/jobs/:id/export` returns 404 for expired jobs (GCS file missing)

No automatic Firestore cleanup - job history preserved indefinitely for audit purposes.

## Client-Side Age Check Pattern

Frontend components should check job age to determine expired state:

```typescript
const isExpired = (job: Job): boolean => {
  const expirationDate = new Date(job.createdAt);
  expirationDate.setDate(expirationDate.getDate() + 7);
  return new Date() > expirationDate;
};

// Hide download button if expired
{!isExpired(job) && job.status === 'complete' && (
  <Button onClick={downloadCSV}>Download Results</Button>
)}

// Show expiration message if expired
{isExpired(job) && (
  <Alert variant="warning">
    Files expired - data deleted after 7 days
  </Alert>
)}
```

This pattern should be applied in:
- Job detail pages
- Job list views (to gray out expired rows)
- Export button components

## Troubleshooting

**Q: Files not deleting after 7 days?**
A: Lifecycle rules take 24 hours to propagate. Check rule is active via `gcloud storage buckets describe`. Note that evaluation runs once per day, so exact timing may vary.

**Q: Can we have different retention for uploads vs results?**
A: Yes, but not currently implemented. Current design uses single 7-day policy for simplicity. To differentiate, add prefix matching to lifecycle rule condition:

```json
{
  "condition": {
    "age": 7,
    "matchesPrefix": ["uploads/"]
  }
}
```

**Q: What happens to Firestore after file deletion?**
A: Nothing. Job documents remain unchanged. Only GCS files are deleted. This allows users to view their job history even after files expire.

**Q: Can users extend retention for specific files?**
A: No. Lifecycle rules apply uniformly to all matching objects. To preserve specific files, they would need to be moved to a different bucket or prefix excluded from the rule.

## Related Requirements

- **OPS-07:** Old uploaded files auto-delete (7-day retention)
- **OPS-08:** Old result files auto-delete (7-day retention)
- **OPS-13:** Signed URLs generated on-demand (15-minute expiration)

## Maintenance

Lifecycle rule requires no ongoing maintenance. Once configured, GCS automatically evaluates and deletes old files.

**To change retention period:**
1. Edit `lifecycle.json` with new age value
2. Reapply with `gcloud storage buckets update` command
3. Wait 24 hours for propagation

**To disable lifecycle rule:**
```bash
gcloud storage buckets update gs://${GCS_BUCKET_NAME} --clear-lifecycle
```

**To view current rule:**
```bash
gcloud storage buckets describe gs://${GCS_BUCKET_NAME} --format="json(lifecycle)"
```

## Security Considerations

- Lifecycle deletion is **permanent** - no soft delete or recovery period
- Files deleted by lifecycle rules do NOT generate notifications to Pub/Sub (unlike manual deletion)
- Ensure Firestore job status reflects GCS state accurately to prevent users attempting to download expired files
- Consider implementing client-side warnings when jobs approach 7-day expiration (e.g., "Files expire in 2 days")

## Cost Impact

7-day retention significantly reduces storage costs compared to indefinite retention:
- Average job: 5MB upload + 2MB results = 7MB total
- 100 jobs/month = 700MB if deleted after 7 days vs 8.4GB after 1 year
- Cost savings: ~$8/year per 100 monthly jobs (at $0.02/GB/month Standard Storage pricing)

For production workloads with 1000+ jobs/month, lifecycle rules are essential for cost control.

---

**Last Updated:** 2026-04-07
**Bucket:** `autoencoder_data`
**Rule Status:** Active (configured 2026-04-07)
