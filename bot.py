import os
import logging
from pyrogram import Client, filters
from pyrogram.types import Message
from src.audio_utils import process_video_audio
from src.asr_utils import chunk_based_transcription_vad, save_transcript_segments
from src.subtitle_utils import generate_srt, generate_dual_srt
from src.make_it_correct import make_it_correct_openai
from src.acent import convert_to_dialect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_ID = "x"
API_HASH = "x" 
BOT_TOKEN = "x:x-x"

UPLOAD_FOLDER = 'data/input_videos'
AUDIO_OUTPUTS = 'data/audio_outputs'
TEXT_OUTPUTS = 'data/text_outputs'
SUBTITLES_FOLDER = 'data/subtitles'
CHUNKS_FOLDER = 'data/chunks'

for folder in [UPLOAD_FOLDER, AUDIO_OUTPUTS, TEXT_OUTPUTS, SUBTITLES_FOLDER, CHUNKS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

pending_jobs = {}

def process_video(job_data):
    try:
        video_path = job_data['video_path']
        filename = job_data['filename']
        filename_base = os.path.splitext(filename)[0]
        original_lang = job_data['original_lang']
        target_lang = job_data['target_lang']
        enhance_audio = job_data['enhance_audio']
        correct_text = job_data['correct_text']
        dialect = job_data['dialect']

        clean_audio_path = os.path.join(AUDIO_OUTPUTS, f"{filename_base}_clean.wav")
        transcript_path = os.path.join(TEXT_OUTPUTS, f"{filename_base}_transcript.txt")
        srt_path = os.path.join(SUBTITLES_FOLDER, f"{filename_base}.srt")
        dual_srt_path = os.path.join(SUBTITLES_FOLDER, f"{filename_base}_dual.srt")

        logger.info(f"پردازش صدا برای {video_path}...")
        process_video_audio(video_path, clean_audio_path, target_sr=16000, enhance=enhance_audio)

        logger.info("اجرای رونویسی با VAD...")
        segments = chunk_based_transcription_vad(clean_audio_path, sr=16000)
        save_transcript_segments(segments, transcript_path)

        logger.info("تولید فایل‌های زیرنویس...")
        generate_srt(segments, srt_path)
        generate_dual_srt(segments, original_lang, target_lang, dual_srt_path)

        if correct_text:
            logger.info("اصلاح زیرنویس...")
            with open("src/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
            with open(srt_path, "r", encoding="utf-8") as f:
                message_srt = f.read()
            with open(dual_srt_path, "r", encoding="utf-8") as f:
                message_dual_srt = f.read()

            result_srt = make_it_correct_openai(message=message_srt, system_prompt=system_prompt)
            result_dual_srt = make_it_correct_openai(message=message_dual_srt, system_prompt=system_prompt)

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(result_srt)
            with open(dual_srt_path, "w", encoding="utf-8") as f:
                f.write(result_dual_srt)

        if dialect != "none":
            logger.info("اعمال تبدیل گویش...")
            with open(srt_path, "r", encoding="utf-8") as f:
                current_srt = f.read()
            with open(dual_srt_path, "r", encoding="utf-8") as f:
                current_dual_srt = f.read()

            converted_srt = convert_to_dialect(dialect, current_srt)
            converted_dual_srt = convert_to_dialect(dialect, current_dual_srt)

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(converted_srt)
            with open(dual_srt_path, "w", encoding="utf-8") as f:
                f.write(converted_dual_srt)

        return {
            'filename': filename,
            'original_lang': original_lang,
            'target_lang': target_lang,
            'transcript_path': transcript_path,
            'srt_path': srt_path,
            'dual_srt_path': dual_srt_path
        }

    except Exception as e:
        logger.error(f"خطا در پردازش: {str(e)}")
        raise

app = Client("video_processor_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

@app.on_message(filters.command("start"))
def start_handler(client: Client, message: Message):
    message.reply_text(
        "سلام! یک فایل ویدیویی برای من ارسال کنید تا صدای آن را استخراج کرده و زیرنویس تولید کنم."
    )

@app.on_message(filters.video | filters.document)
def video_handler(client: Client, message: Message):
    try:
        if message.document:
            original_file_name = message.document.file_name
        elif message.video and hasattr(message.video, "file_name"):
            original_file_name = message.video.file_name
        else:
            original_file_name = "video.mp4"

        download_path = os.path.join(UPLOAD_FOLDER, original_file_name)
        file_path = message.download(file_name=download_path)

        pending_jobs[message.from_user.id] = {
            'job_data': {'video_path': file_path, 'filename': os.path.basename(file_path)},
            'step': 1,
            'params': {}
        }

        message.reply_text(
            "ویدیو دریافت شد! لطفا تنظیمات پردازش را انجام دهید.\n\n"
            "مرحله 1: لطفا زبان اصلی را انتخاب کنید:\n"
            "en - انگلیسی\n"
            "fa - فارسی\n"
            "es - اسپانیایی"
        )

    except Exception as e:
        logger.error(f"خطا در دریافت ویدیو: {str(e)}")
        message.reply_text(f"خطایی رخ داد: {str(e)}")

def parse_bool(text):
    return text.strip().lower() in ('بله', 'yes', 'true', '1')

@app.on_message(filters.text)
def conversation_handler(client: Client, message: Message):
    user_id = message.from_user.id
    if user_id not in pending_jobs:
        return

    state = pending_jobs[user_id]
    current_step = state['step']
    text = message.text.strip()

    if current_step == 1:
        state['params']['original_lang'] = text
        state['step'] = 2
        message.reply_text(
            "مرحله 2: لطفا زبان مقصد را انتخاب کنید:\n"
            "fa - فارسی\n"
            "en - انگلیسی\n"
            "es - اسپانیایی"
        )

    elif current_step == 2:
        state['params']['target_lang'] = text
        state['step'] = 3
        message.reply_text("مرحله 3: آیا می‌خواهید کیفیت صدا بهبود یابد؟ (بله/خیر)")

    elif current_step == 3:
        state['params']['enhance_audio'] = parse_bool(text)
        state['step'] = 4
        message.reply_text("مرحله 4: آیا می‌خواهید زیرنویس‌ها اصلاح شوند؟ (بله/خیر)")

    elif current_step == 4:
        state['params']['correct_text'] = parse_bool(text)
        state['step'] = 5
        message.reply_text(
            "مرحله 5: لطفا گویش مورد نظر را انتخاب کنید:\n"
            "none - بدون تغییر\n"
            "isfahani - اصفهانی\n"
            "shirazi - شیرازی"
        )

    elif current_step == 5:
        state['params']['dialect'] = text
        job_data = state['job_data']
        job_data.update(state['params'])
        pending_jobs.pop(user_id)

        message.reply_text("در حال پردازش ویدیو... لطفا صبر کنید.")

        try:
            results = process_video(job_data)
            reply_message = (
                "پردازش تکمیل شد!\n\n"
                f"فایل متن: {os.path.basename(results['transcript_path'])}\n"
                f"فایل زیرنویس: {os.path.basename(results['srt_path'])}\n"
                f"فایل زیرنویس دوزبانه: {os.path.basename(results['dual_srt_path'])}"
            )
            message.reply_text(reply_message)
            message.reply_document(results['transcript_path'], caption="متن")
            message.reply_document(results['srt_path'], caption="زیرنویس")
            message.reply_document(results['dual_srt_path'], caption="زیرنویس دوزبانه")

        except Exception as e:
            logger.error(f"خطا در پردازش ویدیو: {str(e)}")
            message.reply_text(f"خطایی در پردازش رخ داد: {str(e)}")

if __name__ == "__main__":
    print("ربات در حال اجراست...")
    app.run()
