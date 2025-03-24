import os
import logging
import time
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from src.audio_utils import process_video_audio
from src.asr_utils import chunk_based_transcription_vad, save_transcript_segments
from src.subtitle_utils import generate_srt, generate_dual_srt
from src.make_it_correct import make_it_correct_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = 'data/input_videos'
CHUNKS_FOLDER = 'data/chunks'
OUTPUT_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CHUNKS_FOLDER'] = CHUNKS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

app.jinja_env.cache = {}

for directory in [
    'data/input_videos', 
    'data/audio_outputs', 
    'data/text_outputs', 
    'data/subtitles',
    'data/chunks'
]:
    os.makedirs(directory, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('فایل ویدیو ارسال نشده است.')
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        flash('هیچ فایلی انتخاب نشده است.')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        original_lang = request.form.get('original_lang', 'en')
        target_lang = request.form.get('target_lang', 'fa')
        enhance_audio = 'enhance_audio' in request.form
        correct_text = 'correct_text' in request.form
        dialect = request.form.get('dialect', 'none')
        
        job_data = {
            'video_path': video_path,
            'filename': filename,
            'original_lang': original_lang,
            'target_lang': target_lang,
            'enhance_audio': enhance_audio,
            'correct_text': correct_text,
            'dialect': dialect
        }
        
        return process_video(job_data)
    
    flash('فرمت فایل نامعتبر است. فرمت‌های مجاز: mp4, avi, mov, mkv')
    return redirect(url_for('index'))

@app.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    chunk_number = int(request.form.get('chunk_number'))
    total_chunks = int(request.form.get('total_chunks'))
    filename = secure_filename(request.form.get('filename'))
    file_id = request.form.get('file_id')
    
    file_chunk_dir = os.path.join(app.config['CHUNKS_FOLDER'], file_id)
    os.makedirs(file_chunk_dir, exist_ok=True)
    
    chunk_file = request.files['chunk']
    chunk_path = os.path.join(file_chunk_dir, f"chunk_{chunk_number}")
    chunk_file.save(chunk_path)
    
    if chunk_number == total_chunks - 1:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(video_path, 'wb') as outfile:
            for i in range(total_chunks):
                chunk_path = os.path.join(file_chunk_dir, f"chunk_{i}")
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())
        for i in range(total_chunks):
            chunk_path = os.path.join(file_chunk_dir, f"chunk_{i}")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        os.rmdir(file_chunk_dir)
        
        return jsonify({'status': 'success', 'message': 'آپلود فایل کامل شد', 'filename': filename})
    
    return jsonify({'status': 'success', 'message': f'دریافت chunk {chunk_number + 1}/{total_chunks}'})

@app.route('/process', methods=['POST'])
def process_uploaded():
    filename = request.form.get('filename')
    if not filename or not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        flash('فایل پیدا نشد یا نامعتبر است.')
        return redirect(url_for('index'))
    
    original_lang = request.form.get('original_lang', 'en')
    target_lang = request.form.get('target_lang', 'fa')
    enhance_audio = request.form.get('enhance_audio') == '1'
    correct_text = request.form.get('correct_text') == '1'
    dialect = request.form.get('dialect', 'none')
    
    job_data = {
        'video_path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
        'filename': filename,
        'original_lang': original_lang,
        'target_lang': target_lang,
        'enhance_audio': enhance_audio,
        'correct_text': correct_text,
        'dialect': dialect
    }
    
    return process_video(job_data)

def process_video(job_data):
    try:
        video_path = job_data['video_path']
        filename_base = os.path.splitext(job_data['filename'])[0]
        original_lang = job_data['original_lang']
        target_lang = job_data['target_lang']
        enhance_audio = job_data['enhance_audio']
        correct_text = job_data['correct_text']
        dialect = job_data['dialect']
        
        clean_audio_path = f"data/audio_outputs/{filename_base}_clean.wav"
        transcript_path = f"data/text_outputs/{filename_base}_transcript.txt"
        srt_path = f"data/subtitles/{filename_base}.srt"
        dual_srt_path = f"data/subtitles/{filename_base}_dual.srt"
        
        logger.info(f"پردازش صدا برای {video_path}...")
        process_video_audio(video_path, clean_audio_path, target_sr=16000, enhance=enhance_audio)
        
        logger.info("اجرای رونویسی با VAD...")
        segments = chunk_based_transcription_vad(clean_audio_path, sr=16000)
        save_transcript_segments(segments, transcript_path)
        
        logger.info("تولید فایل‌های زیرنویس...")
        generate_srt(segments, srt_path)
        generate_dual_srt(segments, original_lang, target_lang, dual_srt_path)
        
        if correct_text:
            logger.info("اصلاح زیرنویس با استفاده از make_it_correct_openai...")
            with open("src/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
            message_srt = open(srt_path, "r", encoding="utf-8").read()
            message_dual_srt = open(dual_srt_path, "r", encoding="utf-8").read()
            
            corrected_srt = make_it_correct_openai(message=message_srt, system_prompt=system_prompt)
            corrected_dual_srt = make_it_correct_openai(message=message_dual_srt, system_prompt=system_prompt)
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(corrected_srt)
            with open(dual_srt_path, "w", encoding="utf-8") as f:
                f.write(corrected_dual_srt)
        
        if dialect != "none":
            logger.info("اعمال تبدیل گویش...")
            from src.acent import convert_to_dialect
            current_srt = open(srt_path, "r", encoding="utf-8").read()
            current_dual_srt = open(dual_srt_path, "r", encoding="utf-8").read()
            
            converted_srt = convert_to_dialect(dialect, current_srt)
            converted_dual_srt = convert_to_dialect(dialect, current_dual_srt)
            
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(converted_srt)
            with open(dual_srt_path, "w", encoding="utf-8") as f:
                f.write(converted_dual_srt)
        
        results = {
            'filename': job_data['filename'],
            'original_lang': original_lang,
            'target_lang': target_lang,
            'transcript_path': os.path.basename(transcript_path),
            'srt_path': os.path.basename(srt_path),
            'dual_srt_path': os.path.basename(dual_srt_path)
        }
        
        return render_template('output.html', results=results, current_time=int(time.time()))
    
    except Exception as e:
        logger.error(f"خطا در پردازش: {str(e)}")
        flash(f"خطا در پردازش ویدیو: {str(e)}")
        return redirect(url_for('index'))

@app.route('/download/<path:filename>')
def download_file(filename):
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    if file.endswith('.srt'):
        response = send_from_directory(os.path.join(OUTPUT_FOLDER, directory), file, as_attachment=request.args.get('download', 'false').lower() == 'true')
        response.headers['Content-Type'] = 'application/x-subrip; charset=utf-8'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    return send_from_directory(os.path.join(OUTPUT_FOLDER, directory), file, as_attachment=True)

@app.route('/serve-video/<path:filename>')
def serve_video(filename):
    response = send_from_directory(os.path.join(OUTPUT_FOLDER, 'input_videos'), filename, as_attachment=False)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/update_subtitle', methods=['POST'])
def update_subtitle():
    file_type = request.form.get('file_type', 'srt')
    new_content = request.form.get('new_content')
    file_name = request.form.get('file_name')
    if not file_name:
        return jsonify({'status': 'error', 'message': 'نام فایل مشخص نشده است.'}), 400
    file_path = os.path.join("data/subtitles", file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return jsonify({'status': 'success', 'message': 'زیرنویس به‌روزرسانی شد.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
