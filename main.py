import os
import logging
from src.audio_utils import process_video_audio
from src.asr_utils import chunk_based_transcription_vad, save_transcript_segments
from src.subtitle_utils import generate_srt, generate_dual_srt
from src.acent import convert_to_dialect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    video_path = "data/input_videos/sample1.mp4"
    clean_audio_path = "data/audio_outputs/sample_clean.wav"
    transcript_path = "data/text_outputs/sample_transcript.txt"
    srt_path = "data/subtitles/sample.srt"
    dual_srt_path = "data/subtitles/sample_dual.srt"

    original_lang = input("Enter the video language code (e.g., 'en', 'fa'): ").strip().lower() or "en"
    target_lang = input("Enter the target language code for subtitles (e.g., 'fa'): ").strip().lower() or "fa"
    enhance_input = input("Enhance the audio? (Y/n): ").strip().lower()
    enhance_audio = False if enhance_input in ["n", "no"] else True

    for d in [os.path.dirname(clean_audio_path),
              os.path.dirname(transcript_path),
              os.path.dirname(srt_path),
              os.path.dirname(dual_srt_path)]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    try:
        process_video_audio(video_path, clean_audio_path, target_sr=16000, enhance=enhance_audio)
        logger.info("Phase 1 completed successfully.")

        segments = chunk_based_transcription_vad(clean_audio_path, sr=16000)
        save_transcript_segments(segments, transcript_path)
        logger.info("Phase 2 completed successfully.")

        generate_srt(segments, srt_path)
        generate_dual_srt(segments, original_lang, target_lang, dual_srt_path)
        
        correct_text = input("Do you want to correct the text? (Y/n): ").strip().lower()
        if correct_text != 'n':
            print("Correcting the text...")
            message_srt = open("data/subtitles/sample.srt", mode="r", encoding="utf-8").read()
            message_dual_srt = open("data/subtitles/sample_dual.srt", mode="r", encoding="utf-8").read()
            with open("src/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
            from src.make_it_correct import make_it_correct_openai
            result_srt = make_it_correct_openai(message=message_srt, system_prompt=system_prompt)
            result_dual_srt = make_it_correct_openai(message=message_dual_srt, system_prompt=system_prompt)
            open("data/subtitles/sample.srt", "w", encoding="utf-8").write(result_srt)
            open("data/subtitles/sample_dual.srt", "w", encoding="utf-8").write(result_dual_srt)
        
        accent_correction = input("Do you want to apply accent correction to the transcript? (y/n): ").strip().lower()
        if accent_correction == 'y':
            dialect = input("Enter the dialect for accent correction (e.g., 'shirazi', 'isfahani'): ").strip().lower()
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            corrected_text = convert_to_dialect(dialect, transcript_text)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)
            logger.info("Accent correction applied successfully.")

        logger.info("Phase 3 completed successfully.")
    except Exception as e:
        logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
