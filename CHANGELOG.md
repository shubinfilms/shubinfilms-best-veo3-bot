# Changelog

## v0.7.0-freeze-suno
- Freeze current bot release prior to Suno reliability improvements.
- Harden Suno enqueue retries with jittered exponential backoff and 12s cap.
- Ensure Suno refunds and failure messaging are idempotent per request ID.
- Add regression coverage for retry timing and failure deduplication.
