# UI Button Architecture

This bot centralises all interactive buttons under the `ui.buttons` package. Each button
has a declarative contract describing the screen or dialog that must open. Handlers stay
small and composable, while the router provides idempotency, telemetry and safety rails.

## Key concepts

- **Registry** – `ui.buttons.registry.BUTTONS` is the single source of truth. Every entry
  includes:
  - `id` – stable identifier used in callback payloads and metrics.
  - `title_i18n_key` – localisation key for captions.
  - `access` – required access tiers (`all`, `paid`, `admin`).
  - `handler` – async callable returning a `UIResult` contract.
  - `open_telemetry_event` – event name for distributed traces.
  - `feature_flag` – optional gate for staged rollouts.
- **Router** – `ui.buttons.router.dispatch` validates flags, enforces access, applies a
  Redis-backed idempotency lock (`chat_id + button_id`), emits Prometheus metrics and wraps
  handler failures into soft, user-facing errors. The router also performs an immediate
  callback acknowledgement (to dismiss Telegram's loading spinner) and debounces repeated
  clicks from the same user for a short window (default 400 ms).
- **Results** – handlers return a `UIResult` describing the expected UI. Tests assert
  against the result instead of raw Telegram side-effects.
- **Idempotency** – `ui.buttons.idempotency.with_idempotency` rejects duplicate clicks
  while the first handler execution is in-flight. A lightweight in-process debounce window
  shields the handlers from button chatter (rapid tapping) and records telemetry in
  `ui_callback_dedup_total`.

## Adding a new button

1. Implement a handler in `ui/buttons/handlers.py`. Reuse `_open_menu_item` when the button
   opens a legacy menu card and return the appropriate helper from `ui.buttons.results`
   (`show_menu`, `show_dialog`, etc.). Document the contract in a docstring.
2. Register the button in `ui/buttons/registry.py` with the desired access tier and feature
   flag. This registry is the only place where button metadata lives.
3. Extend `ui/buttons/guards.py` if the handler needs new access tiers or feature signals.
4. Add unit tests that call `ui.buttons.router.dispatch` to cover success, feature-flagged
   and access-denied scenarios. Use the returned `UIResult` for assertions.
5. Wire existing callback handlers (see `handle_main_menu_callback`) to `dispatch` so every
   entry point flows through the router.

## Testing matrix

- **Unit tests**: `tests/test_button_router.py` validates registry coverage, access checks
  and feature flag behaviour.
- **Integration tests**: exercise real Telegram updates that call `dispatch` and assert the
  returned contracts.
- **E2E tests**: cover the `Button × User state` matrix to guarantee the declared contract
  opens the expected screen and remains idempotent.

## Contracts

The declarative `UIResult` makes button behaviour auditable. When introducing a new button,
write down the expected UI in the handler docstring and mirror it in tests. Contracts should
avoid hidden side effects so the CI matrix can reliably detect regressions.
