# Phase 2: Worker Reliability - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Make async job processing robust: handle duplicate Pub/Sub messages, prevent race conditions in Firestore status updates, extend ack deadlines for long-running Vertex AI jobs (10-15 min), and validate arbitrary CSV formats without crashing or wasting Vertex AI quota.

</domain>

<decisions>
## Implementation Decisions

### Ack deadline & timing
- **Ack timing**: Acknowledge Pub/Sub messages only after processing completes successfully (not before processing starts)
- **Ack extension mechanism**: Use background thread (threading.Timer) to extend ack deadline periodically (every 60 seconds) during long-running jobs
- **Extension failure handling**: Log warning and continue processing (rely on idempotency handling to prevent duplicate jobs if message is redelivered)
- **Extension interval**: Hardcoded 60-second interval (sufficient for v1 with 10-15 min jobs, configurable later if needed)

### Claude's Discretion
- Message field validation approach (schema validation library vs manual checks)
- Idempotency tracking implementation (Firestore document structure, TTL strategy)
- Firestore transaction patterns (optimistic vs pessimistic locking)
- Status transition state machine design (allowed transitions, error states)
- CSV validation specifics (encoding detection, structure checks, size limits)
- Defense-in-depth validation strategy (Express vs Worker layer responsibilities)
- Edge case handling for CSVs (unicode, mixed types, mostly-missing values, wide datasets)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `google.cloud.pubsub_v1.SubscriberClient`: Already used for message subscription (worker.py:273)
- `google.cloud.firestore.Client`: Already initialized for status tracking (worker.py:49)
- `validate_environment()`: Existing pattern for fail-fast env var checks (worker.py:52-69) — reusable for new config
- `logging` module: Already configured (worker.py:46-47) — extend for ack extension warnings

### Established Patterns
- Pub/Sub callback pattern: `callback(message)` function receives messages (worker.py:236-259)
- Firestore status updates: `db.collection('jobs').document(job_id).set({...}, merge=True)` (worker.py:79-80, 171-176)
- Error handling: Try/catch blocks with Firestore error status writes (worker.py:180-185, 224-229)
- Message format: `{jobId, bucket, file}` JSON payload (worker.py:242-246)

### Integration Points
- **callback() function** (worker.py:236): Entry point for all message processing — add field validation here
- **process_upload_local()** (worker.py:72): Local processing flow — add ack extension here
- **process_upload_vertex()** (worker.py:188): Vertex AI dispatch — add ack extension here
- **Firestore status writes** (lines 79, 171, 182, 226): Convert to transactions to prevent race conditions
- **message.ack()** (line 253): Move to after successful processing, not before

### Current Reliability Gaps (from code review)
- **No message field validation**: Lines 244-246 extract `jobId`, `bucket`, `file` without checking if they exist
- **Premature ack**: Line 253 acknowledges message immediately after processing starts, not after completion
- **No idempotency tracking**: Duplicate Pub/Sub message delivery will trigger duplicate Vertex AI jobs
- **No ack deadline extension**: Long-running jobs (10-15 min) exceed default ack deadline, causing redelivery
- **Firestore updates without transactions**: Concurrent status updates (e.g., worker writes "processing" while Vertex AI job completes) can create race conditions
- **No CSV validation before processing**: Encoding errors, malformed structure discovered only after training starts

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard async processing patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-worker-reliability*
*Context gathered: 2026-04-03*
