# Suno Music Generation Runbook

This runbook summarises the operational flow of the Suno integration in the bot and
highlights the moving parts that now guarantee late deliveries.

## Startup & Database

* The ledger backend connects to PostgreSQL during startup (`ledger._prepare`).
  All DDL statements are idempotent and safe to re-run on PG14/PG15.
* Postgres connection pools are opened explicitly (`psycopg_pool.AsyncConnectionPool.open`),
  eliminating the deprecated implicit open warning.

## Generation Lifecycle

1. **Enqueue** – `_poll_suno_and_send` stores task metadata through `SunoService`.
2. **Polling** – `SunoService.poll_record_info_once` uses the configured
   `SUNO_TASK_STATUS_PATH`. A `404` response is treated as `pending`.
3. **Timeout** – After `SUNO_POLL_TIMEOUT_SEC` the poller emits a timeout,
   notifies the user once, and issues a refund (`suno:refund:timeout`).
4. **Delivery** – When tracks are ready, `_suno_deliver_record_info_payload` forwards
   them through `SunoService.handle_callback` (via webhook, poll, or reconciliation).

## Reconciliation Worker

* Controlled by environment variables:
  * `SUNO_RECONCILE_INTERVAL_SEC` (default `180s`)
  * `SUNO_RECONCILE_MAX_AGE_SEC` (default `24h`)
  * `SUNO_RECONCILE_BATCH` (default `40`)
* Runs in the background once the bot is up (`_suno_reconcile_worker`).
* Scans recent task records (Redis/in-memory) and re-queries the status endpoint.
* Uses `_SUNO_RECONCILE_ATTEMPTS` to throttle per-task checks and avoids hammering
  tasks still in progress.
* Late successes are delivered with `delivery_via="reconcile"` so Telegram users
  still receive their tracks even after a timeout refund.

## Refund Policy

* Refunds are issued only for terminal API states (FAILED/ERROR/EXPIRED) or when
  the poller times out. `404`/`pending` states do **not** trigger refunds.
* Reconciliation does not attempt to refund; it only handles deliveries.

## Troubleshooting Checklist

1. **No delivery & no refund** – Inspect Redis keys `suno:pending:*` and
   `_SUNO_RECONCILE_ATTEMPTS` to ensure reconciliation is running.
2. **Repeated refunds** – Confirm the task status is actually terminal by
   checking the recorded payload in `SUNO_SERVICE.get_task_record`.
3. **Status endpoint mismatch** – Validate `SUNO_TASK_STATUS_PATH` in the
   deployment environment; logs now print the resolved path at startup.

## Useful Commands

* Run focused tests:

  ```bash
  pytest tests/test_suno_poll_pending.py tests/test_suno_reconcile.py
  ```

* Inspect last stored tasks (within a REPL):

  ```python
  import bot
  bot.SUNO_SERVICE.list_last_tasks(limit=5)
  ```
