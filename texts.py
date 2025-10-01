from __future__ import annotations

from typing import Any

from suno.cover_source import MAX_AUDIO_MB

FAQ_INTRO = "🧾 *FAQ*\nВыберите раздел:"

FAQ_SECTIONS = {
    "veo": "🎬 *Видео (VEO)*\n• Fast — быстрее и дешевле.\n• Quality — дольше, но лучше детализация.\n• Формат: присылаете идею/ фото → карточка → «Сгенерировать».\n• Время: 2–10 мин.",
    "mj": "🎨 *Изображения (MJ)*\n• Стоимость: 10💎 за 1 изображение.\n• Один бесплатный перезапуск при сетевой ошибке.",
    "banana": "🧩 *Banana (редактор)*\n• До 4 фото, затем текст: фон, одежда, макияж, удаление объектов, объединение людей.\n• Время: 1–5 мин.",
    "suno": "🎵 *Музыка (Suno)*\n• Введите тему/настроение и длительность.\n• Текст песни можно сгенерировать в Prompt-Master.",
    "billing": "💎 *Баланс и оплата*\n• Пополнение через Stars в меню.\n• Где купить Stars: PremiumBot.\n• Баланс: /my_balance.",
    "tokens": "⚡ *Токены и возвраты*\n• Списываются при старте.\n• При ошибке/таймауте бот возвращает 💎 автоматически.",
    "chat": "💬 *Обычный чат*\n• /chat включает режим, /reset очищает контекст.\n• Поддерживаются голосовые — бот расшифрует.",
    "pm": "🧠 *Prompt-Master*\n• Помогает быстро получить качественный промпт.\n• Кнопки категорий в самом Prompt-Master.",
    "common": "ℹ️ *Общие вопросы*\n• Куда приходят клипы/изображения: прямо в чат.\n• Если бот «молчит»: проверьте баланс и повторите запрос.",
}

SUNO_RU = {
    "suno.mode.cover": "Ковер",
    "suno.mode.instrumental": "Инструментал",
    "suno.mode.vocal": "Музыка с вокалом",
    "suno.field.title": "Название",
    "suno.field.style": "Стиль",
    "suno.field.lyrics": "Текст",
    "suno.field.lyrics_source": "Источник текста",
    "suno.field.source": "Источник",
    "suno.field.cost": "Стоимость",
    "suno.lyrics_source.user": "🧾 Мой текст",
    "suno.lyrics_source.ai": "✨ Сгенерировать ИИ",
    "suno.prompt.mode_select": "Выберите режим генерации",
    "suno.prompt.step.title": (
        "Введите короткое название трека. Отправьте /cancel, чтобы остановить.\n"
        "Сейчас: “{current}”"
    ),
    "suno.prompt.step.style": (
        "Шаг {index}/{total} (стиль): Опишите стиль/теги (например, «эмбиент, мягкие барабаны»). "
        "Отправьте /cancel, чтобы остановить."
    ),
    "suno.prompt.step.lyrics": (
        "Шаг {index}/{total} (текст): Пришлите текст песни одним сообщением. "
        "Отправьте /cancel, чтобы остановить."
    ),
    "suno.prompt.step.source": (
        f"Шаг {{index}}/{{total}} (источник): Пришлите аудио-файл (mp3/wav, до {MAX_AUDIO_MB} МБ) "
        "или ссылку на аудио (http/https)."
    ),
    "suno.prompt.step.generic": "🎯 Уточните следующий параметр.",
    "suno.prompt.fill": "Заполните: {fields}",
    "suno.prompt.ready": "Все обязательные поля заполнены. Можно запускать генерацию.",
    "suno.prompt.starting": "Запускаю генерацию…",
    "suno.error.upload_client": (
        "⚠️ Не удалось принять аудио. Пришлите mp3/wav (до {MAX_AUDIO_MB} МБ) или ссылку http/https."
    ),
    "suno.error.upload_service": (
        "⚠️ Не удалось принять аудио. Пришлите mp3/wav (до {MAX_AUDIO_MB} МБ) или ссылку http/https."
    ),
}


def t(key: str, /, **kwargs: Any) -> str:
    value = SUNO_RU.get(key, key)
    if kwargs:
        try:
            return value.format(**kwargs)
        except Exception:
            return value
    return value


SUNO_MODE_PROMPT = t("suno.prompt.mode_select")
SUNO_START_READY_MESSAGE = t("suno.prompt.ready")
SUNO_STARTING_MESSAGE = t("suno.prompt.starting")
