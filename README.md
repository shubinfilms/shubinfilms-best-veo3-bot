# Best VEO3 Bot (Fast)

## Env (Render → Environment)
- TELEGRAM_TOKEN      = <токен бота>
- OPENAI_API_KEY      = <ключ OpenAI>           # для Prompt-Master (можно пусто — тогда PM отключён)
- KIE_API_KEY         = <ключ KIE>              # просто ключ; код сам добавит 'Bearer '
- KIE_BASE_URL        = https://api.kie.ai
- KIE_VEO_GEN_PATH    = /api/v1/veo/generate
- KIE_VEO_STATUS_PATH = /api/v1/veo/recordInfo
- KIE_VEO_1080_PATH   = /api/v1/veo/get1080pVideo
- KIE_MJ_GENERATE     = /api/v1/mj/generate
- KIE_MJ_STATUS       = /api/v1/mj/recordInfo
- KIE_ENABLE_FALLBACK = false                   # true только если надо включать Fallback при 16:9
- KIE_DEFAULT_SEED    =                          # опционально: число 10000–99999
- KIE_WATERMARK_TEXT  =                          # опционально

## Запуск локально
pip install -r requirements.txt
python bot.py

## Midjourney интерактивный поток
1. Нажмите кнопку «Генерация изображений (MJ)» в главном меню — откроется карточка с выбором формата (16:9 или 9:16, по умолчанию горизонтальный 16:9).
2. Выберите формат. Бот запомнит выбор и покажет карточку «Введите промпт…» с кнопками «Подтвердить», «Отменить», «Сменить формат».
3. Отправьте текст сообщением и нажмите «Подтвердить». Если промпта нет, бот напомнит «❌ Промпт не найден…».
4. После запуска бот ждёт ответ KIE и присылает изображение с подписью и кнопками «Открыть», «Повторить», «Назад в меню». Кнопка «Повторить» оставляет формат и просит новый промпт.
