# Best VEO3 Bot (Fast)

## Env (Render → Environment)
- TELEGRAM_TOKEN      = <токен бота>
- OPENAI_API_KEY      = <ключ OpenAI>           # для Prompt-Master (можно пусто — тогда PM отключён)
- KIE_API_KEY         = <ключ KIE>              # просто ключ; код сам добавит 'Bearer '
- KIE_BASE_URL        = https://api.kie.ai
- KIE_GEN_PATH        = /api/v1/veo/generate
- KIE_STATUS_PATH     = /api/v1/veo/record-info
- KIE_HD_PATH         = /api/v1/veo/get-1080p-video
- KIE_ENABLE_FALLBACK = false                   # true только если надо включать Fallback при 16:9
- KIE_DEFAULT_SEED    =                          # опционально: число 10000–99999
- KIE_WATERMARK_TEXT  =                          # опционально

## Запуск локально
pip install -r requirements.txt
python bot.py
