# Changelog

## Unreleased
- Add localized `/help` and `/support` handler with support chat button and telemetry.
- Document `SUPPORT_USERNAME` and `SUPPORT_USER_ID` environment variables.

## v0.7.1-suno-stable
- Prefix Suno request IDs with the initiating user ID for idempotent reuse.
- Expand enqueue retry jitter to ±30% and align max delay with 15s budget.
- Persist pending metadata under the expanded request ID format in tests.
- Render VEO/MJ prompt cards with empty placeholders until users type.
- Add regression coverage for request ID reuse and duplicate suppression.

## v0.7.0-freeze-suno
- Freeze current bot release prior to Suno reliability improvements.
- Harden Suno enqueue retries with jittered exponential backoff and 12s cap.
- Ensure Suno refunds and failure messaging are idempotent per request ID.
- Add regression coverage for retry timing and failure deduplication.
