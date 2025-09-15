# Best VEO3 Bot (Fast)

## Env (Render → Environment)
Скопируйте файл `.env.пример` в `.env` и заполните значения. Все переменные должны совпадать с тем, что читает `bot.py`.

### Telegram / базовые настройки
- `TELEGRAM_TOKEN` — токен бота.
- `PROMPTS_CHANNEL_URL` — ссылка на канал с промптами (по умолчанию `https://t.me/bestveo3promts`).
- `STARS_BUY_URL` — ссылка на покупку звёзд (по умолчанию `https://t.me/PremiumBot`).
- `PROMO_ENABLED` — включить промокоды (`true` / `false`).
- `DEV_MODE` — режим разработки (`true` / `false`).

### Prompt-Master / OpenAI
- `OPENAI_API_KEY` — ключ OpenAI, можно оставить пустым, чтобы отключить Prompt-Master.

### KIE API
- `KIE_API_KEY` — ключ KIE (без приставки `Bearer`).
- `KIE_BASE_URL` — базовый URL (`https://api.kie.ai`).
- `KIE_VEO_GEN_PATH` — путь создания VEO (`/api/v1/veo/generate`).
- `KIE_VEO_STATUS_PATH` — путь проверки статуса VEO (`/api/v1/veo/recordInfo`).
- `KIE_VEO_1080_PATH` — путь получения 1080p видео (`/api/v1/veo/get1080pVideo`).
- `KIE_MJ_GENERATE` — путь создания MJ (`/api/v1/mj/generate`).
- `KIE_MJ_STATUS` — путь проверки статуса MJ (`/api/v1/mj/recordInfo`).

### Обработка видео
- `FFMPEG_BIN` — путь до `ffmpeg` (по умолчанию `ffmpeg`).
- `ENABLE_VERTICAL_NORMALIZE` — нормализация вертикальных видео (`true` / `false`).
- `ALWAYS_FORCE_FHD` — принудительный 1080p (`true` / `false`).
- `MAX_TG_VIDEO_MB` — ограничение размера видео в МБ (по умолчанию `48`).
- `POLL_INTERVAL_SECS` — интервал опроса задач (в секундах, по умолчанию `6`).
- `POLL_TIMEOUT_SECS` — таймаут ожидания (в секундах, по умолчанию `1200`).

### Логирование и хранение
- `LOG_LEVEL` — уровень логирования (по умолчанию `INFO`).
- `REDIS_URL` — URL Redis (если пусто, Redis отключён).
- `REDIS_PREFIX` — префикс ключей Redis (по умолчанию `veo3:prod`).
- `BALANCE_BACKUP_PATH` — путь к файлу резервного хранения балансов (по умолчанию рядом с `bot.py`).

## Запуск локально
```bash
pip install -r requirements.txt
python bot.py
```
