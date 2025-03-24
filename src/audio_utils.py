import av
import numpy as np
import librosa
import soundfile as sf
import logging
import noisereduce as nr
import pyloudnorm as pyln
from scipy import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_audio_from_video(video_path: str):
    try:
        container = av.open(video_path)
    except Exception as e:
        logger.error(f"Error opening video {video_path}: {e}")
        raise e

    audio_stream = None
    for stream in container.streams:
        if stream.type == 'audio':
            audio_stream = stream
            break
    if audio_stream is None:
        raise ValueError("No audio stream found in the video.")

    audio_frames = []
    sample_rate = audio_stream.rate if hasattr(audio_stream, "rate") else 48000
    try:
        for frame in container.decode(audio_stream):
            pcm = frame.to_ndarray(format='flt')
            if pcm.ndim > 1:
                pcm = np.mean(pcm, axis=0)
            audio_frames.append(pcm)
    except Exception as e:
        logger.error(f"Error decoding audio frames: {e}")
        raise e

    if not audio_frames:
        raise ValueError("No audio frames decoded.")
    audio_data = np.concatenate(audio_frames, axis=0)
    logger.info("Audio extraction from video completed successfully.")
    return audio_data, sample_rate

def speech_enhance(audio_data: np.ndarray, sr: int) -> np.ndarray:
    logger.info("Starting speech enhancement...")
    nyquist = 0.5 * sr
    low = 250 / nyquist
    high = 4000 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    filtered_audio = 0.7 * filtered_audio + 0.3 * audio_data
    percentile_val = np.percentile(np.abs(filtered_audio), 10)
    noise_mask = np.abs(filtered_audio) < percentile_val * 1.5
    if np.sum(noise_mask) > sr * 0.1:
        noise_sample = filtered_audio[noise_mask]
    else:
        noise_sample = filtered_audio[:int(sr * 0.2)]
    reduced_noise_audio = nr.reduce_noise(
        y=filtered_audio,
        sr=sr,
        y_noise=noise_sample,
        stationary=False,
        prop_decrease=0.25,
        n_fft=1024,
        win_length=512,
        n_jobs=1,
        thresh_n_mult_nonstationary=1.5,
    )
    def adaptive_compress(audio, threshold=-20, ratio=1.5):
        threshold_linear = 10 ** (threshold / 20)
        output = np.zeros_like(audio)
        for i, sample in enumerate(audio):
            if abs(sample) > threshold_linear:
                if sample > 0:
                    output[i] = threshold_linear + (sample - threshold_linear) / ratio
                else:
                    output[i] = -threshold_linear + (sample + threshold_linear) / ratio
            else:
                output[i] = sample
        return output
    compressed_audio = adaptive_compress(reduced_noise_audio)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(compressed_audio)
    target_loudness = -18.0
    normalized_audio = pyln.normalize.loudness(compressed_audio, loudness, target_loudness)
    if np.max(np.abs(normalized_audio)) > 0.99:
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
    logger.info("Speech enhancement completed.")
    return normalized_audio

def reduce_noise_and_enhance(audio_data: np.ndarray, sr: int) -> np.ndarray:
    enhanced_audio = speech_enhance(audio_data, sr)
    def deess(audio, sr):
        nyquist = 0.5 * sr
        high_freq = 5000 / nyquist
        ess_b, ess_a = signal.butter(3, high_freq, btype='highpass')
        ess_band = signal.filtfilt(ess_b, ess_a, audio)
        ess_band_compressed = np.tanh(ess_band * 0.8) / 0.8
        ess_energy = np.sqrt(np.mean(ess_band**2))
        compressed_energy = np.sqrt(np.mean(ess_band_compressed**2))
        if compressed_energy > 0:
            gain = ess_energy / compressed_energy
            ess_band_compressed *= gain
        low_freq = 4800 / nyquist
        low_b, low_a = signal.butter(3, low_freq, btype='lowpass')
        low_band = signal.filtfilt(low_b, low_a, audio)
        result = low_band + ess_band_compressed * 0.7
        return result
    deessed_audio = deess(enhanced_audio, sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(deessed_audio)
    target_loudness = -18.0
    final_audio = pyln.normalize.loudness(deessed_audio, loudness, target_loudness)
    logger.info("Enhanced audio processing completed.")
    return final_audio

def save_audio_to_wav(audio_data: np.ndarray, sr: int, output_path: str):
    try:
        sf.write(output_path, audio_data, sr)
        logger.info(f"Audio saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving audio to {output_path}: {e}")
        raise e

def process_video_audio(video_path: str, output_path: str, target_sr: int = 16000, enhance: bool = True):
    logger.info(f"Starting processing for video: {video_path}")
    raw_audio, orig_sr = extract_audio_from_video(video_path)
    logger.info(f"Resampling audio from {orig_sr} Hz to {target_sr} Hz...")
    audio_data = librosa.resample(raw_audio, 
                                orig_sr=orig_sr, 
                                target_sr=target_sr,
                                res_type='kaiser_best')

    if enhance:
        processed_audio = reduce_noise_and_enhance(audio_data, target_sr)
        orig_output_path = output_path.replace(".wav", "_original.wav")
        save_audio_to_wav(audio_data, target_sr, orig_output_path)
        logger.info(f"Original resampled audio saved to {orig_output_path} for comparison.")
    else:
        logger.info("Enhancement disabled; using resampled audio without further processing.")
        processed_audio = audio_data

    save_audio_to_wav(processed_audio, target_sr, output_path)
    logger.info("Phase 1: Audio processing completed.")