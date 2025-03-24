import logging
from typing import List, Tuple, Dict
from deep_translator import GoogleTranslator
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_srt_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def smart_segment_text(text: str, max_chars: int = 42) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    
    segments = []
    break_chars = ['. ', '! ', '? ', '; ', ', ', ' ']
    
    while len(text) > max_chars:
        best_break_idx = -1
        
        for char in break_chars:
            slice_to_check = text[max_chars//2:max_chars]
            pos = slice_to_check.find(char)
            
            if pos != -1:
                best_break_idx = pos + max_chars//2 + len(char) - 1
                break
        
        if best_break_idx == -1:
            best_break_idx = max_chars
            while best_break_idx < len(text) and text[best_break_idx] != ' ':
                best_break_idx -= 1
            if best_break_idx <= max_chars // 2:
                best_break_idx = max_chars
        segments.append(text[:best_break_idx+1].strip())
        text = text[best_break_idx+1:].strip()
    if text:
        segments.append(text)
    
    return segments

def optimize_subtitle_timing(words: List[Tuple[float, float, str]], 
                           min_duration: float = 1.0, 
                           max_duration: float = 7.0, 
                           max_chars_per_line: int = 42) -> List[Tuple[float, float, str]]:
    if not words:
        return []
    
    full_text = " ".join(word[2] for word in words)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    start_time = words[0][0]
    end_time = words[-1][1]
    
    duration = end_time - start_time
    
    if duration < min_duration:
        end_time = start_time + min_duration
    
    if len(full_text) <= max_chars_per_line and duration <= max_duration:
        return [(start_time, end_time, full_text)]
    
    text_segments = smart_segment_text(full_text, max_chars_per_line)
    
    result = []
    segment_count = len(text_segments)
    
    word_to_time = {}
    for word_start, word_end, word_text in words:
        for word in word_text.split():
            word = word.strip().lower()
            if word not in word_to_time and word:
                word_to_time[word] = (word_start, word_end)
    
    current_position = 0
    for i, segment in enumerate(text_segments):
        segment_words = segment.lower().split()
        
        if not segment_words:
            continue
            
        if i == 0:
            seg_start = start_time
        else:
            for word in segment_words:
                if word in word_to_time:
                    seg_start = word_to_time[word][0]
                    break
            else:
                seg_start = start_time + (duration * current_position / len(full_text))
        
        if i == segment_count - 1:
            seg_end = end_time
        else:
            for word in reversed(segment_words):
                if word in word_to_time:
                    seg_end = word_to_time[word][1]
                    break
            else:
                current_position += len(segment)
                seg_end = start_time + (duration * current_position / len(full_text))
        
        if seg_end - seg_start < min_duration:
            seg_end = seg_start + min_duration
        
        result.append((seg_start, seg_end, segment))
    
    return result

def generate_srt(segments: List[Tuple[float, float, str, List[Tuple[float, float, str]]]], srt_output: str):
    try:
        with open(srt_output, "w", encoding="utf-8") as f:
            subtitle_index = 1
            
            for segment_start, segment_end, text, words in segments:
                if not words:
                    subtitles = [(segment_start, segment_end, text)]
                else:
                    subtitles = optimize_subtitle_timing(words)
                
                for start, end, subtitle_text in subtitles:
                    start_ts = format_srt_timestamp(start)
                    end_ts = format_srt_timestamp(end)
                    
                    f.write(f"{subtitle_index}\n{start_ts} --> {end_ts}\n{subtitle_text.strip()}\n\n")
                    subtitle_index += 1
                    
        logger.info(f"SRT file saved to {srt_output}.")
    except Exception as e:
        logger.error(f"Error generating SRT: {e}")
        raise

def generate_dual_srt(segments: List[Tuple[float, float, str, List[Tuple[float, float, str]]]], original_lang: str, target_lang: str, srt_output: str):
    translator = GoogleTranslator(source=original_lang, target=target_lang)
    
    context_window = []
    grouped_segments = []
    
    for i, segment in enumerate(segments):
        start, end, text, words = segment
        
        if i == 0 or start - segments[i-1][1] > 2.0:
            if context_window:
                grouped_segments.append(context_window)
                context_window = []
            
            context_window = [segment]
        else:
            context_window.append(segment)
    
    if context_window:
        grouped_segments.append(context_window)
    
    try:
        with open(srt_output, "w", encoding="utf-8") as f:
            subtitle_index = 1
            
            for group in grouped_segments:
                combined_text = " ".join(segment[2] for segment in group)
                
                try:
                    translated_full = translator.translate(combined_text)
                    
                    orig_lengths = [len(segment[2]) for segment in group]
                    total_orig_len = sum(orig_lengths)
                    trans_approx_positions = []
                    
                    pos = 0
                    for length in orig_lengths[:-1]:
                        ratio = length / total_orig_len
                        pos += int(ratio * len(translated_full))
                        trans_approx_positions.append(pos)
                    
                    translated_segments = []
                    last_pos = 0
                    
                    for pos in trans_approx_positions:
                        break_chars = ['. ', '! ', '? ', '; ', ', ', ' ']
                        best_pos = pos
                        
                        window = 10
                        for char in break_chars:
                            window_text = translated_full[max(0, pos-window):min(len(translated_full), pos+window)]
                            char_pos = window_text.find(char)
                            
                            if char_pos != -1:
                                best_pos = max(0, pos-window) + char_pos + len(char)
                                break
                        
                        translated_segments.append(translated_full[last_pos:best_pos].strip())
                        last_pos = best_pos
                    
                    translated_segments.append(translated_full[last_pos:].strip())
                    
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    translated_segments = []
                    for segment in group:
                        try:
                            translated = translator.translate(segment[2])
                            translated_segments.append(translated)
                        except:
                            translated_segments.append("[Translation Error]")
                
                for i, (segment, translated) in enumerate(zip(group, translated_segments)):
                    segment_start, segment_end, text, words = segment
                    
                    if not words:
                        subtitles = [(segment_start, segment_end, translated)]
                    else:
                        orig_subtitles = optimize_subtitle_timing(words)
                        
                        trans_lines = smart_segment_text(translated, max_chars=42)
                        
                        if len(orig_subtitles) == len(trans_lines):
                            subtitles = [(start, end, trans) 
                                        for (start, end, _), trans in zip(orig_subtitles, trans_lines)]
                        else:
                            if len(orig_subtitles) > 0:
                                if len(trans_lines) == 1:
                                    subtitles = [(start, end, translated) for start, end, _ in orig_subtitles]
                                else:
                                    ratio = len(trans_lines) / len(orig_subtitles)
                                    subtitles = []
                                    
                                    for j, (start, end, _) in enumerate(orig_subtitles):
                                        start_idx = min(int(j * ratio), len(trans_lines) - 1)
                                        end_idx = min(int((j + 1) * ratio), len(trans_lines))
                                        
                                        if start_idx < end_idx:
                                            combined_text = " ".join(trans_lines[start_idx:end_idx])
                                            subtitles.append((start, end, combined_text))
                                        else:
                                            subtitles.append((start, end, trans_lines[min(j, len(trans_lines)-1)]))
                            else:
                                subtitles = [(segment_start, segment_end, translated)]
                    
                    for start, end, trans_text in subtitles:
                        start_ts = format_srt_timestamp(start)
                        end_ts = format_srt_timestamp(end)
                        
                        f.write(f"{subtitle_index}\n{start_ts} --> {end_ts}\n{trans_text.strip()}\n\n")
                        subtitle_index += 1
                    
        logger.info(f"Translated SRT saved to {srt_output}.")
    except Exception as e:
        logger.error(f"Error generating translated SRT: {e}")
        raise