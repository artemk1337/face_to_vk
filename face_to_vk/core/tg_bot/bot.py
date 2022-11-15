import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ConversationHandler, ApplicationBuilder, ContextTypes, CommandHandler

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

PHOTO = 0


async def is_admin(id: int) -> bool:
    # Проверяет пользователя на админа
    return True


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет, дружище! Я помогу тебе найти ту самую девушку из метро. Напишите /photo_search, чтобы загрузить фотографию.")


async def start_photo_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Пришлите мне фотографию человека, которого вы хотите найти"
    )
    return PHOTO


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """We get a photo from the user and perform a search."""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    logger.info("Фото содержит в себе %s", photo_file)
    logger.info("Фотография %s: получено", user.first_name)
    # await photo_file.download("user_photo.jpg")
    # logger.info("Photo of %s: %s", user.first_name, "user_photo.jpg")
    await update.message.reply_text(
        "Подождите, идет поиск."
    )
    # ТУТ НАДО ПЕРЕДАТЬ ФОТО ДАЛЬШЕ
    await update.message.reply_text(
        "Я нашел похожих людей \n"
        "1)что-то 1\n"
        "2)что-то 2"
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("Пользователь %s отменил разговор.", user.first_name)
    await update.message.reply_text("Выполнено")

    return ConversationHandler.END


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm sorry, but you wrote nonsense")


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display a help message"""
    if (is_admin(update.message.from_user.id)):
        # help admin
        await update.message.reply_text("Доступные команды админа:\n(Отсутвуют)")
    else:
        # help user
        await update.message.reply_text(
            "Используйте /start чтобы начать общение. Используйте /cancel чтобы очистить историю общения.")


def main() -> None:
    application = ApplicationBuilder().token('5709512510:AAHbbXhmQdpTNIHC0ms0e_LIML-2ODWDyiE').build()

    start_handler = CommandHandler('start', start)
    photo_search_handler = ConversationHandler(
        entry_points=[CommandHandler("photo_search", start_photo_search)],
        states={
            PHOTO: [MessageHandler(filters.PHOTO, photo)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    # Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application.add_handler(start_handler)
    application.add_handler(photo_search_handler)
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(unknown_handler)
    application.add_handler(echo_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
