import logging
import os
import librosa
import numpy as np
import tempfile
import soundfile as sf
from typing import List, Tuple
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_asr_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading HF pipeline with openai/whisper-large-v3-turbo model.")
    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        return_timestamps=True,
        chunk_length_s=30,
        device=device
    )
    return asr_pipe

def vad_segmentation_energy(audio: np.ndarray, sr: int, frame_duration_ms: int = 30, energy_threshold_factor: float = 0.1, merge_threshold: float = 0.5) -> List[Tuple[float, float, np.ndarray]]:
    frame_length = int(sr * frame_duration_ms / 1000)
    num_frames = int(np.ceil(len(audio) / frame_length))
    rms_values = []
    for i in range(num_frames):
        frame = audio[i * frame_length:min((i + 1) * frame_length, len(audio))]
        if len(frame) > 0:
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
        else:
            rms_values.append(0)
    rms_values = np.array(rms_values)
    sorted_rms = np.sort(rms_values)
    noise_level = np.percentile(sorted_rms, 10)
    median_rms = np.median(sorted_rms)
    energy_threshold = noise_level + energy_threshold_factor * (median_rms - noise_level)
    logger.info(f"Adaptive energy threshold set to: {energy_threshold:.6f}")
    segments = []
    segment_start = None
    for i, rms in enumerate(rms_values):
        time = i * frame_duration_ms / 1000.0
        if rms > energy_threshold and segment_start is None:
            segment_start = time
        elif rms <= energy_threshold and segment_start is not None:
            segment_end = time
            if segment_end - segment_start > 0.2:
                segments.append((segment_start, segment_end))
            segment_start = None
    if segment_start is not None:
        segments.append((segment_start, num_frames * frame_duration_ms / 1000.0))
    merged_segments = []
    if segments:
        cur_start, cur_end = segments[0]
        for start, end in segments[1:]:
            if start - cur_end < merge_threshold:
                cur_end = end
            else:
                merged_segments.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged_segments.append((cur_start, cur_end))
    extended_segments = []
    padding = 0.1
    for start, end in merged_segments:
        ext_start = max(0, start - padding)
        ext_end = min(len(audio) / sr, end + padding)
        extended_segments.append((ext_start, ext_end))
    segments_info = [(start, end, audio[int(start * sr):int(end * sr)]) for (start, end) in extended_segments]
    logger.info(f"Energy-based VAD produced {len(segments_info)} segments.")
    return segments_info

def chunk_based_transcription_vad(audio_path: str, sr: int = 16000) -> List[Tuple[float, float, str, List[Tuple[float, float, str]]]]:
    asr_pipe = load_asr_pipeline()
    audio, file_sr = librosa.load(audio_path, sr=sr, mono=True)
    segments = vad_segmentation_energy(audio, sr, frame_duration_ms=30, energy_threshold_factor=0.1, merge_threshold=0.5)
    segments_info = []
    for (start, end, seg_audio) in segments:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            seg_path = tmp_wav.name
        sf.write(seg_path, seg_audio, sr, format="WAV")
        try:
            result = asr_pipe(seg_path)
            if isinstance(result, dict) and "text" in result and "chunks" in result:
                text = result["text"].strip()
                words = [(w["timestamp"][0] + start, w["timestamp"][1] + start, w["text"]) for w in result["chunks"]]
            else:
                text = "[Transcription Error]"
                words = []
        except Exception as e:
            logger.error(f"Error transcribing segment from {start:.2f} to {end:.2f}: {e}")
            text = "[Transcription Error]"
            words = []
        os.remove(seg_path)
        if text and not text.isspace() and text != ".":
            segments_info.append((start, end, text, words))
    logger.info("VAD-based transcription with word-level timestamps completed.")
    return segments_info

def save_transcript_segments(segments: List[Tuple[float, float, str, List[Tuple[float, float, str]]]], output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for (start, end, text, words) in segments:
                f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
                for w_start, w_end, w_text in words:
                    f.write(f"    ({w_start:.2f} - {w_end:.2f}): {w_text}\n")
        logger.info(f"Segmented transcript saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving segmented transcript: {e}")
        raise