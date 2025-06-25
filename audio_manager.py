# audio_manager.py
import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import shutil
import random
import re
import textwrap
from typing import Optional, List
import librosa
import soundfile as sf
import av
from av.audio.resampler import AudioResampler
from config import DEPARTMENT_AUDIO_ROOT_PATH
from scene_generator import SceneGenerator

class AudioManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_color = (255, 255, 255)
        self.line_type = 2
        
    def _get_category_folder_name(self, display_name: str) -> str:
        name = re.sub(r'[^a-zA-Z0-9\s]', '', display_name).lower()
        return re.sub(r'\s+', '_', name)

    def _split_into_phrases(self, text: str, max_words: int = 6) -> List[str]:
        words = text.split()
        if not words: return [""]
        phrases = []
        for i in range(0, len(words), max_words):
            phrases.append(" ".join(words[i:i+max_words]))
        return phrases

    def add_subtitles_to_video_opencv(self, video_path: str, prompt_text: str, output_path: str) -> bool:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): st.error(f"Error opening video: {video_path}"); return False
            width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0 or total_frames == 0: st.error("Video has zero frames/FPS."); cap.release(); return False
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            phrases = self._split_into_phrases(prompt_text)
            frames_per_phrase = total_frames // len(phrases)
            subtitle_map = [(i * frames_per_phrase, (i + 1) * frames_per_phrase if i < len(phrases) - 1 else total_frames, phrase) for i, phrase in enumerate(phrases)]
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                active_phrase = ""
                for start_f, end_f, phrase in subtitle_map:
                    if start_f <= frame_idx < end_f: active_phrase = phrase; break
                if active_phrase:
                    wrapped_lines = textwrap.wrap(active_phrase, width=40)
                    y_start = height - 40 - (len(wrapped_lines) - 1) * 35
                    for i, line in enumerate(wrapped_lines):
                        (tw, th), bl = cv2.getTextSize(line, self.font, self.font_scale, self.line_type)
                        line_x, line_y = (width - tw) // 2, y_start + (i * 35)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (line_x - 10, line_y - th - 10), (line_x + tw + 10, line_y + bl), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        cv2.putText(frame, line, (line_x, line_y), self.font, self.font_scale, self.font_color, self.line_type)
                out.write(frame)
                frame_idx += 1
            cap.release(), out.release()
            return True
        except Exception as e:
            st.error(f"Error adding subtitles with OpenCV: {e}"); return False

    def add_audio_to_video_av(self, video_path: str, audio_path: str, output_path: str) -> Optional[str]:
        st.info(f"Processing audio: {os.path.basename(audio_path)}...")
        temp_output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        temp_wav_path = os.path.join(self.temp_dir, "temp_audio.wav")
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            sf.write(temp_wav_path, audio_data.T, sample_rate)
            st.info("Merging with 'av' library...")
            with av.open(video_path) as in_video, av.open(temp_wav_path) as in_audio, av.open(temp_output_file, mode='w') as output:
                in_v_stream, in_a_stream = in_video.streams.video[0], in_audio.streams.audio[0]
                video_duration = float(in_v_stream.duration * in_v_stream.time_base) if in_v_stream.duration else float('inf')
                out_v_stream = output.add_stream("libx264", rate=in_v_stream.average_rate)
                out_v_stream.width, out_v_stream.height, out_v_stream.pix_fmt = in_v_stream.width, in_v_stream.height, "yuv420p"
                out_a_stream = output.add_stream("aac", rate=44100, layout='stereo')
                resampler = AudioResampler(format='fltp', layout='stereo', rate=44100)
                for frame in in_video.decode(in_v_stream):
                    for packet in out_v_stream.encode(frame): output.mux(packet)
                for frame in in_audio.decode(in_a_stream):
                    if (frame.pts * in_a_stream.time_base if frame.pts else 0) > video_duration: break
                    for resampled_frame in resampler.resample(frame):
                        for packet in out_a_stream.encode(resampled_frame): output.mux(packet)
                for packet in out_v_stream.encode(None): output.mux(packet)
                for packet in out_a_stream.encode(None): output.mux(packet)
            shutil.move(temp_output_file, output_path)
            st.success("Music added and merged successfully!")
            return output_path
        except Exception as e:
            st.error(f"Audio processing (PyAV) failed: {e}"); return None
        finally:
            if os.path.exists(temp_output_file): os.remove(temp_output_file)
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)

    def add_department_audio(self, video_path: str, output_dir: str, selected_category: str, locked_audio_path: Optional[str] = None) -> Optional[str]:
        full_audio_path = locked_audio_path
        if not full_audio_path:
            category_folder = self._get_category_folder_name(selected_category)
            audio_dir = os.path.join(DEPARTMENT_AUDIO_ROOT_PATH, category_folder)
            if not os.path.isdir(audio_dir): st.error(f"Audio directory not found: '{audio_dir}'"); return None
            audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.mp3')]
            if not audio_files: st.error(f"No .mp3 files found in '{audio_dir}'"); return None
            full_audio_path = os.path.join(audio_dir, random.choice(audio_files))
            st.session_state.locked_audio_path = full_audio_path
        st.info(f"Using audio: `{os.path.basename(full_audio_path)}`")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_with_audio.mp4")
        return self.add_audio_to_video_av(video_path, full_audio_path, output_path)

    def add_summarized_subtitles(self, video_path: str, output_dir: str, current_prompt: str, scene_generator: SceneGenerator) -> Optional[str]:
        summarized_text = scene_generator.summarize_prompt_for_subtitles(current_prompt)
        st.info(f"Generated Subtitle: \"{summarized_text}\"")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_subtitled.mp4")
        return output_path if self.add_subtitles_to_video_opencv(video_path, summarized_text, output_path) else None

    def add_audio_and_subtitles(self, video_path: str, output_dir: str, selected_category: str, current_prompt: str, scene_generator: SceneGenerator, locked_audio_path: Optional[str] = None) -> Optional[str]:
        summarized_text = scene_generator.summarize_prompt_for_subtitles(current_prompt)
        subtitled_video_path = os.path.join(self.temp_dir, "subtitled.mp4")
        if not self.add_subtitles_to_video_opencv(video_path, summarized_text, subtitled_video_path):
            st.error("Failed to create subtitled video. Aborting."); return None
        full_audio_path = locked_audio_path
        if not full_audio_path:
            category_folder = self._get_category_folder_name(selected_category)
            audio_dir = os.path.join(DEPARTMENT_AUDIO_ROOT_PATH, category_folder)
            if not os.path.isdir(audio_dir): st.error(f"Audio directory not found: '{audio_dir}'"); return None
            audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.mp3')]
            if not audio_files: st.error(f"No .mp3 files in '{audio_dir}'"); return None
            full_audio_path = os.path.join(audio_dir, random.choice(audio_files))
            st.session_state.locked_audio_path = full_audio_path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_final.mp4")
        return self.add_audio_to_video_av(subtitled_video_path, full_audio_path, output_path)