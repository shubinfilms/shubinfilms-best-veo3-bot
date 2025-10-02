# Example environment variables for Best VEO3 Bot
# Copy this file to `.env` and fill in the secrets before running locally.

# Telegram bot credentials
TELEGRAM_TOKEN=your-telegram-token
PROMPTS_CHANNEL_URL=https://t.me/bestveo3promts
STARS_BUY_URL=https://t.me/PremiumBot
PROMO_ENABLED=true
DEV_MODE=false
SUPPORT_USERNAME=BestAi_Support
SUPPORT_USER_ID=7223448532

# OpenAI / Prompt Master (optional)
OPENAI_API_KEY=

## Support

- Команды `/help` и `/support` отправляют локализованное сообщение с кнопкой, которая ведёт в чат поддержки `https://t.me/BestAi_Support` (значение формируется из `SUPPORT_USERNAME`).
- Переменные окружения `SUPPORT_USERNAME` и `SUPPORT_USER_ID` позволяют быстро сменить контакт поддержки без изменений в кодовой базе.

# KIE API configuration
KIE_API_KEY=
KIE_BASE_URL=https://api.kie.ai
KIE_VEO_GEN_PATH=/api/v1/veo/generate
KIE_VEO_STATUS_PATH=/api/v1/veo/record-info
KIE_VEO_1080_PATH=/api/v1/veo/get-1080p-video
KIE_MJ_GENERATE=/api/v1/mj/generate
KIE_MJ_STATUS=/api/v1/mj/recordInfo
KIE_BANANA_MODEL=google/nano-banana-edit

# Suno API
SUNO_ENABLED=true
SUNO_API_BASE=https://api.kie.ai
SUNO_API_PREFIX=
SUNO_GEN_PATH=/suno-api/generate
SUNO_STATUS_PATH=/suno-api/record-info
SUNO_EXTEND_PATH=/suno-api/generate/extend
SUNO_LYRICS_PATH=/suno-api/generate/get-timestamped-lyrics

# Дополнительные KIE настройки (опционально)
KIE_ENABLE_FALLBACK=false         # true только если надо включать fallback при 16:9
KIE_DEFAULT_SEED=                 # опционально: число 10000–99999
KIE_WATERMARK_TEXT=               # опционально

# Video processing
FFMPEG_BIN=ffmpeg
ENABLE_VERTICAL_NORMALIZE=true
ALWAYS_FORCE_FHD=true
MAX_TG_VIDEO_MB=48
POLL_INTERVAL_SECS=6
POLL_TIMEOUT_SECS=1200

# Logging
LOG_LEVEL=INFO

# Redis cache / promo codes / runner lock
REDIS_LOCK_ENABLED=true
REDIS_URL=redis://:password@host:port/0
REDIS_PREFIX=veo3:prod

# Postgres ledger storage (обязательно)
DATABASE_URL=postgresql://user:password@host:5432/database

## Logging
- Управление уровнем логов: `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL` (по умолчанию `INFO`).
- Небуферизованный stdout: `PYTHONUNBUFFERED=1` включён кодом.

## Tests
Локально:
```bash
pytest -q
```

> **Note:** Python 3.13 удалил модуль `imghdr` (PEP 594). Проект использует
> Pillow для определения типа обложек и совместим как с Python 3.12, так и с
> 3.13+.

Проверка callback вручную:

```bash
curl -i -X POST "https://<service>.onrender.com/suno-callback" \
  -H 'Content-Type: application/json' \
  -H 'X-Callback-Secret: <YOUR_SUNO_CALLBACK_SECRET>' \
  -d '{
    "code":200,
    "msg":"ok",
    "data":{"callbackType":"complete","task_id":"demo123",
      "data":[{"id":"trk1","title":"Demo",
               "audio_url":"https://example.com/file.mp3",
               "image_url":"https://example.com/cover.png"}]}}'
```

---

## Render env
- `LOG_LEVEL=INFO` *(временно можно `DEBUG` для диагностики)*
- `SUNO_CALLBACK_SECRET=<secret>` *(уже есть)*

---

## Acceptance Criteria
- При `LOG_LEVEL=INFO` нет спама от `httpx/urllib3/uvicorn/telegram` в логах.
- `/` ⇒ `{"ok": true}`; `/healthz` ⇒ `{"ok": true}`; `/suno-callback` с валидным токеном ⇒ `{"status":"received"}`.
- `pytest -q` проходит локально.
- Ручной `curl` с `complete`-payload:
  - логирует «callback received» и «processed | {...}»,
  - при 403 на аудио — фиксирует `audio-link:...` без падения,
  - при 200 на обложку — сохраняет файл во временной папке и логирует размер,
  - при пустых `TELEGRAM_TOKEN/ADMIN_IDS` пишет «skip Telegram notify».

## Redis runner lock mechanics

* При запуске бот ставит ключ `{REDIS_PREFIX}:lock:runner` в Redis (`SET NX EX=60`). Значение — JSON с `host`, `pid`, `started_at`, `heartbeat_at`, `version`.
* Каждые ~25 секунд происходит heartbeat: TTL продлевается до 60 секунд и обновляется `heartbeat_at`. В логах появляются события `LOCK_HEARTBEAT`.
* Если при старте обнаружен свежий ключ, логируется `LOCK_BUSY`, бот завершается без повторных попыток (чтобы избежать 409 от Telegram).
* Если ключ устарел (нет heartbeat >90 секунд), выводится `LOCK_STALE_TAKEOVER`, старый ключ удаляется и ставится новый.
* При штатном завершении или сигнале SIGINT/SIGTERM ключ удаляется (`LOCK_RELEASED`).
* Чтобы отключить блокировку (например, локально), установите `REDIS_LOCK_ENABLED=false` — Redis для локера тогда не используется.

# --- Midjourney интерактивный поток ---
# 1. В главном меню нажмите «Генерация изображений (MJ)» — появится карточка Midjourney с выбором формата (по умолчанию 16:9).
# 2. Выберите «Горизонтальный (16:9)» или «Вертикальный (9:16)». Выбор сохраняется в профиле, кнопка «Назад» возвращает в меню.
# 3. После выбора формата бот показывает карточку «Введите промпт…» с кнопками «Подтвердить», «Отменить», «Сменить формат».
# 4. Отправьте текст и нажмите «Подтвердить». Если промпт сохранён — бот ответит «✅ Промпт принят». Если текста нет — «❌ Промпт не найден, отправьте текст и повторите».
# 5. Когда генерация завершится, бот пришлёт фото с подписью (формат и первые 100 символов промпта) и кнопками «Открыть», «Повторить», «Назад в меню».
