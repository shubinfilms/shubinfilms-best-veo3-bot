# Example environment variables for Best VEO3 Bot
# Copy this file to `.env` and fill in the secrets before running locally.

# Telegram bot credentials
TELEGRAM_TOKEN=your-telegram-token
PROMPTS_CHANNEL_URL=https://t.me/bestveo3promts
STARS_BUY_URL=https://t.me/PremiumBot
PROMO_ENABLED=true
DEV_MODE=false

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

# Redis cache / promo codes (optional)
REDIS_URL=redis://:password@host:port/0
REDIS_PREFIX=veo3:prod

# --- Midjourney интерактивный поток ---
# 1. Нажмите «Генерация изображений (MJ)» в меню → откроется карточка с выбором формата (16:9 или 9:16).
# 2. Выберите формат. Бот запомнит выбор и покажет карточку «Введите промпт…» с кнопками «Подтвердить», «Отменить», «Сменить формат».
# 3. Отправьте текст и нажмите «Подтвердить». Если промпта нет, бот напомнит «❌ Промпт не найден…».
# 4. После генерации бот пришлёт изображение с кнопками «Открыть», «Повторить», «Назад в меню». Кнопка «Повторить» сохраняет формат.
