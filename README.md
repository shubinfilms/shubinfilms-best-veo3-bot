# Best VEO3 Bot (Fast)

## Env (Render → Environment)
- TELEGRAM_TOKEN      = <токен бота>
- PROMPTS_CHANNEL_URL = https://t.me/bestveo3promts            # ссылка на канал с промптами
- STARS_BUY_URL       = https://t.me/PremiumBot                # куда отправлять за покупкой звёзд
- PROMO_ENABLED       = true                                   # включить промокоды
- DEV_MODE            = false                                  # отладочный режим (меньше логики)

- OPENAI_API_KEY      = <ключ OpenAI>                          # опционально, для Prompt-Master

- KIE_API_KEY         = <ключ KIE>                             # просто ключ; код сам добавит 'Bearer '
- KIE_BASE_URL        = https://api.kie.ai
- KIE_VEO_GEN_PATH    = /api/v1/veo/generate
- KIE_VEO_STATUS_PATH = /api/v1/veo/recordInfo
- KIE_VEO_1080_PATH   = /api/v1/veo/get1080pVideo
- KIE_MJ_GENERATE     = /api/v1/mj/generate                    # опционально: MJ
- KIE_MJ_STATUS       = /api/v1/mj/recordInfo                  # опционально: MJ
- KIE_BANANA_MODEL    = google/nano-banana-edit                # опционально: Banana

- FFMPEG_BIN                = ffmpeg                           # путь к ffmpeg
- ENABLE_VERTICAL_NORMALIZE = true                             # выравнивание вертикального видео
- ALWAYS_FORCE_FHD          = true                             # всегда 1080p
- MAX_TG_VIDEO_MB           = 48                               # предел размера загружаемого видео
- POLL_INTERVAL_SECS        = 6                                # проверка задач KIE, секунд
- POLL_TIMEOUT_SECS         = 1200                             # общий таймаут ожидания

- LOG_LEVEL = INFO

- REDIS_URL    = redis://:password@host:port/0                 # опционально: кеш и промокоды
- REDIS_PREFIX = veo3:prod

## Запуск локально
pip install -r requirements.txt
python bot.py
