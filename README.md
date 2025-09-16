# Example environment variables for Best VEO3 Bot
# Copy this file to `.env` and fill in the secrets before running locally.

# Telegram bot credentials
TELEGRAM_TOKEN=your-telegram-token
PROMPTS_CHANNEL_URL=https://t.me/bestveo3promts
STARS_BUY_URL=https://t.me/PremiumBot
PROMO_ENABLED=true
MENU_COMPACT=false
PROMO_CODES=WELCOME50=50,FREE10=10              # опционально: список CODE=AMOUNT через запятую или перенос строки
PROMO_CODES_JSON={"SPRING200": 200}            # опционально: JSON-словарь с промокодами
PROMO_CODES_FILE=/app/config/promo_codes.txt    # опционально: путь до файла с промокодами (JSON или CODE=AMOUNT)
DEV_MODE=false
ADMIN_ID=123456789                              # опционально: ID администратора для /promolist

# OpenAI / Prompt Master (optional)
OPENAI_API_KEY=

# KIE API configuration
KIE_API_KEY=
KIE_BASE_URL=https://api.kie.ai
KIE_VEO_GEN_PATH=/api/v1/veo/generate
KIE_VEO_STATUS_PATH=/api/v1/veo/record-info
KIE_VEO_1080_PATH=/api/v1/veo/get-1080p-video
KIE_MJ_GENERATE=/api/v1/mj/generate
KIE_MJ_STATUS=/api/v1/mj/recordInfo
KIE_BANANA_MODEL=google/nano-banana-edit

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
