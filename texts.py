from __future__ import annotations

from typing import Any, Optional

from suno.cover_source import MAX_AUDIO_MB

FAQ_INTRO = "🧾 *FAQ*\nВыберите раздел:"

TXT_MENU_TITLE = "📋 Главное меню"
TXT_PROFILE_TITLE = "👤 Профиль"
TXT_KB_PROFILE = "👤 Профиль"
TXT_KB_KNOWLEDGE = "📚 База знаний"
TXT_KB_PHOTO = "📸 Режим фото"
TXT_KB_MUSIC = "🎧 Режим музыки"
TXT_KB_VIDEO = "📹 Режим видео"
TXT_KB_AI_DIALOG = "🧠 Диалог с ИИ"
TXT_TOPUP_ENTRY = "💎 Пополнить баланс"
TXT_TOPUP_CHOOSE = "Оплатить с помощью:"
TXT_PAY_STARS = "⭐️ Телеграм Stars"
TXT_PAY_CARD = "💳 Оплата картой"
TXT_PAY_CRYPTO = "🔐 Crypto"
TXT_CRYPTO_COMING_SOON = "Крипто-оплата скоро будет доступна."
TXT_PAY_CRYPTO_OPEN_LINK = "Открыть оплату в браузере"
TXT_AI_DIALOG_NORMAL = "💬 Обычный чат"
TXT_AI_DIALOG_PM = "📝 Prompt-Master"
TXT_AI_DIALOG_CHOOSE = "Выберите режим диалога:"
TXT_KNOWLEDGE_INTRO = "📚 База знаний\nВыберите раздел:"

COMMON_TEXTS_RU = {
    "topup.menu.title": "Выберите способ пополнения:",
    "topup.menu.stars": "💎 Оплатить звёздами",
    "topup.menu.yookassa": "💳 Оплатить картой (ЮKassa)",
    "topup.menu.back": "⬅️ Назад",
    "topup.inline.open": "💳 Пополнить баланс",
    "topup.inline.back": "⬅️ Назад в меню",
    "topup.yookassa.pack_1": "Пакет 1 (+X1💎)",
    "topup.yookassa.pack_2": "Пакет 2 (+X2💎)",
    "topup.yookassa.pack_3": "Пакет 3 (+X3💎)",
    "topup.yookassa.title": "Выберите пакет пополнения:",
    "topup.yookassa.pay": "Перейти к оплате",
    "topup.yookassa.created": "Счёт создан. Перейдите к оплате:",
    "topup.yookassa.retry": "Попробовать снова",
    "topup.yookassa.error": "⚠️ Не удалось создать платёж. Попробуйте ещё раз.",
    "topup.yookassa.processing": "⚠️ Обработка платежа уже идёт. Подождите пару секунд и обновите меню.",
    "topup.stars.title": "💎 Пополнение через Telegram Stars",
    "topup.stars.info": (
        "Если звёзд не хватает — купите в официальном боте @PremiumBot."
    ),
    "balance.insufficient": "Недостаточно токенов: нужно {need}💎, на балансе {have}💎.",
    "balance.success": "Оплата прошла успешно! Баланс: {new_balance}💎.",
}

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

HELP_I18N = {
    "ru": {
        "title": "🆘 Поддержка",
        "body": (
            "Напишите нам, если что-то не работает, есть идея или нужен совет.\n"
            "Ответим как можно скорее.\n\n"
            "• Чат поддержки: @{support_username}\n"
            "• Язык: автоматически — по языку профиля Telegram"
        ),
        "button": "Написать в поддержку",
    },
    "en": {
        "title": "🆘 Support",
        "body": (
            "Message us if something breaks, you have an idea, or need guidance.\n"
            "We’ll reply as soon as possible.\n\n"
            "• Support chat: @{support_username}\n"
            "• Language: auto — from your Telegram profile"
        ),
        "button": "Message Support",
    },
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
        "Шаг {index}/{total} (название): Введите короткое название трека. "
        "Отправьте /cancel, чтобы остановить.\n"
        "Сейчас: “{current}”"
    ),
    "suno.prompt.step.style": (
        "Шаг {index}/{total} (стиль): Опишите стиль/теги (например, „эмбиент, мягкие барабаны“). "
        "Отправьте /cancel, чтобы остановить.\n"
        "Сейчас: “{current}”"
    ),
    "suno.prompt.step.lyrics": (
        "Шаг {index}/{total} (текст): Пришлите текст песни (до {limit} символов) или отправьте /skip, "
        "чтобы сгенерировать автоматически.\n"
        "Сейчас: “{current}”"
    ),
    "suno.prompt.step.source": (
        f"Шаг {{index}}/{{total}} (источник): Пришлите аудио-файл (mp3/wav, до {MAX_AUDIO_MB} МБ) "
        "или ссылку на аудио (http/https)."
    ),
    "suno.prompt.step.generic": "🎯 Уточните следующий параметр.",
    "suno.prompt.fill": "Заполните: {fields}",
    "suno.prompt.ready": "Все обязательные поля заполнены. Можно запускать генерацию.",
    "suno.prompt.starting": "Запускаю генерацию…",
    "suno.error.upload_client": "⚠️ Не удалось загрузить источник. Проверьте файл/ссылку и попробуйте ещё раз.",
    "suno.error.upload_service": "⚠️ Сервис загрузки недоступен. Попробуйте позже.",
}


def t(key: str, /, **kwargs: Any) -> str:
    value = SUNO_RU.get(key, key)
    if kwargs:
        try:
            return value.format(**kwargs)
        except Exception:
            return value
    return value


def help_text(language_code: Optional[str], support_username: str) -> tuple[str, str]:
    """Return localized help message text and button label."""

    locale = "ru"
    if isinstance(language_code, str) and language_code:
        lowered = language_code.lower()
        if lowered.startswith("en"):
            locale = "en"
    data = HELP_I18N.get(locale, HELP_I18N["ru"])
    body = data["body"].format(support_username=support_username)
    return f"{data['title']}\n\n{body}", data["button"]


SUNO_MODE_PROMPT = t("suno.prompt.mode_select")
SUNO_START_READY_MESSAGE = t("suno.prompt.ready")
SUNO_STARTING_MESSAGE = t("suno.prompt.starting")


def common_text(key: str, /, **kwargs: Any) -> str:
    value = COMMON_TEXTS_RU.get(key, key)
    if kwargs:
        try:
            return value.format(**kwargs)
        except Exception:
            return value
    return value
