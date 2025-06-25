import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datetime import datetime, timedelta
import re
import math

class SubtitleManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Subtitle styling options
        self.font_sizes = {
            'small': 18,
            'medium': 24,
            'large': 32,
            'extra_large': 40
        }
        
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255)
        }
        
        self.bg_colors = {
            'transparent': (0, 0, 0, 0),
            'black_semi': (0, 0, 0, 128),
            'white_semi': (255, 255, 255, 128),
            'black_solid': (0, 0, 0, 255),
            'white_solid': (255, 255, 255, 255)
        }
        
        self.positions = {
            'bottom': 'bottom',
            'top': 'top',
            'center': 'center',
            'bottom_left': 'bottom_left',
            'bottom_right': 'bottom_right',
            'top_left': 'top_left',
            'top_right': 'top_right'
        }
        
        # Default settings
        self.default_settings = {
            'font_size': 'medium',
            'text_color': 'white',
            'bg_color': 'black_semi',
            'position': 'bottom',
            'words_per_second': 2.5,
            'max_chars_per_line': 50,
            'max_lines': 2,
            'fade_duration': 0.3,
            'margin': 50
        }
    
    def parse_srt_format(self, srt_content: str) -> List[Dict]:
        """Parse SRT subtitle format"""
        try:
            subtitles = []
            blocks = srt_content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Parse subtitle number
                    subtitle_num = int(lines[0])
                    
                    # Parse time range
                    time_line = lines[1]
                    time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
                    
                    if time_match:
                        start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
                        end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])
                        
                        start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
                        end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
                        
                        # Parse text (can be multiple lines)
                        text = '\n'.join(lines[2:])
                        
                        subtitles.append({
                            'id': subtitle_num,
                            'start': start_time,
                            'end': end_time,
                            'text': text,
                            'duration': end_time - start_time
                        })
            
            return subtitles
            
        except Exception as e:
            st.error(f"Error parsing SRT format: {str(e)}")
            return []
    
    def generate_auto_subtitles(self, text: str, video_duration: float, 
                               words_per_second: float = 2.5) -> List[Dict]:
        """Generate automatic subtitle timing from text"""
        try:
            # Clean and split text into sentences
            text = re.sub(r'\s+', ' ', text.strip())
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            subtitles = []
            current_time = 0.0
            subtitle_id = 1
            
            for sentence in sentences:
                if current_time >= video_duration:
                    break
                
                words = sentence.split()
                if not words:
                    continue
                
                # Calculate duration based on word count and reading speed
                word_count = len(words)
                duration = word_count / words_per_second
                
                # Apply constraints
                duration = max(1.0, min(duration, 6.0))  # Between 1-6 seconds
                
                # Adjust if it exceeds video duration
                if current_time + duration > video_duration:
                    duration = video_duration - current_time
                
                # Break long sentences into multiple subtitles
                if len(sentence) > self.default_settings['max_chars_per_line'] * self.default_settings['max_lines']:
                    chunks = self.split_long_sentence(sentence)
                    chunk_duration = duration / len(chunks)
                    
                    for chunk in chunks:
                        if current_time >= video_duration:
                            break
                        
                        subtitles.append({
                            'id': subtitle_id,
                            'start': current_time,
                            'end': min(current_time + chunk_duration, video_duration),
                            'text': chunk,
                            'duration': min(chunk_duration, video_duration - current_time)
                        })
                        
                        current_time += chunk_duration
                        subtitle_id += 1
                else:
                    subtitles.append({
                        'id': subtitle_id,
                        'start': current_time,
                        'end': min(current_time + duration, video_duration),
                        'text': sentence,
                        'duration': min(duration, video_duration - current_time)
                    })
                    
                    current_time += duration
                    subtitle_id += 1
            
            return subtitles
            
        except Exception as e:
            st.error(f"Error generating auto subtitles: {str(e)}")
            return []
    
    def split_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentences into chunks that fit subtitle constraints"""
        max_chars = self.default_settings['max_chars_per_line']
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length <= max_chars:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_subtitle_frame(self, text: str, frame_width: int, frame_height: int, 
                             settings: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create subtitle overlay for a single frame"""
        try:
            if settings is None:
                settings = self.default_settings.copy()
            
            # Create transparent image
            img = Image.new('RGBA', (frame_width, frame_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Load font
            font_size = self.font_sizes[settings['font_size']]
            try:
                # Try to load system fonts
                font_paths = [
                    "arial.ttf", "Arial.ttf", "calibri.ttf", "Calibri.ttf",
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "C:/Windows/Fonts/arial.ttf"  # Windows
                ]
                
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except:
                font = ImageFont.load_default()
            
            # Prepare text
            wrapped_text = self.wrap_text(text, settings['max_chars_per_line'])
            lines = wrapped_text.split('\n')
            
            # Limit number of lines
            if len(lines) > settings['max_lines']:
                lines = lines[:settings['max_lines']]
                if len(lines) == settings['max_lines']:
                    lines[-1] += "..."
            
            # Calculate text dimensions
            line_height = font_size + 5
            total_text_height = len(lines) * line_height
            
            # Calculate position
            x_pos, y_pos = self.calculate_position(
                frame_width, frame_height, total_text_height, 
                settings['position'], settings['margin']
            )
            
            # Get colors
            text_color = self.colors[settings['text_color']]
            bg_color = self.bg_colors[settings['bg_color']]
            
            # Draw each line
            for i, line in enumerate(lines):
                # Calculate line position
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                line_x = x_pos if settings['position'] in ['top_left', 'bottom_left'] else (frame_width - text_width) // 2
                line_y = y_pos + (i * line_height)
                
                # Draw background if not transparent
                if bg_color[3] > 0:  # Has alpha
                    padding = 5
                    bg_rect = [
                        line_x - padding, line_y - 2,
                        line_x + text_width + padding, line_y + text_height + 2
                    ]
                    draw.rectangle(bg_rect, fill=bg_color)
                
                # Draw text
                draw.text((line_x, line_y), line, font=font, fill=text_color)
            
            # Convert to numpy arrays
            img_array = np.array(img)
            
            # Separate RGB and alpha channels
            if img_array.shape[2] == 4:
                rgb_array = img_array[:, :, :3]
                alpha_array = img_array[:, :, 3] / 255.0
            else:
                rgb_array = img_array
                alpha_array = np.ones((frame_height, frame_width), dtype=np.float32)
            
            # Convert RGB to BGR for OpenCV
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            return bgr_array, alpha_array
            
        except Exception as e:
            st.error(f"Error creating subtitle frame: {str(e)}")
            # Return empty arrays
            empty_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            empty_alpha = np.zeros((frame_height, frame_width), dtype=np.float32)
            return empty_frame, empty_alpha
    
    def wrap_text(self, text: str, max_chars: int) -> str:
        """Wrap text to fit within character limit"""
        return textwrap.fill(text, width=max_chars)
    
    def calculate_position(self, frame_width: int, frame_height: int, text_height: int, 
                          position: str, margin: int) -> Tuple[int, int]:
        """Calculate subtitle position based on settings"""
        positions = {
            'bottom': (margin, frame_height - text_height - margin),
            'top': (margin, margin),
            'center': (margin, (frame_height - text_height) // 2),
            'bottom_left': (margin, frame_height - text_height - margin),
            'bottom_right': (frame_width - margin, frame_height - text_height - margin),
            'top_left': (margin, margin),
            'top_right': (frame_width - margin, margin)
        }
        
        return positions.get(position, positions['bottom'])
    
    def apply_fade_effect(self, alpha: np.ndarray, current_time: float, 
                         start_time: float, end_time: float, fade_duration: float) -> np.ndarray:
        """Apply fade in/out effect to subtitle"""
        try:
            fade_alpha = 1.0
            
            # Fade in
            if current_time < start_time + fade_duration:
                fade_progress = (current_time - start_time) / fade_duration
                fade_alpha = max(0.0, min(1.0, fade_progress))
            
            # Fade out
            elif current_time > end_time - fade_duration:
                fade_progress = (end_time - current_time) / fade_duration
                fade_alpha = max(0.0, min(1.0, fade_progress))
            
            return alpha * fade_alpha
            
        except Exception as e:
            st.error(f"Error applying fade effect: {str(e)}")
            return alpha
    
    def add_subtitles_to_video(self, video_path: str, subtitles: List[Dict], 
                              output_path: str, settings: Dict = None) -> bool:
        """Add subtitles to video"""
        try:
            if settings is None:
                settings = self.default_settings.copy()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error opening video file")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                st.error("Error creating output video file")
                cap.release()
                return False
            
            frame_idx = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_idx / fps
                
                # Find active subtitles
                active_subtitles = []
                for subtitle in subtitles:
                    if subtitle['start'] <= current_time <= subtitle['end']:
                        active_subtitles.append(subtitle)
                
                # Apply subtitles
                for subtitle in active_subtitles:
                    subtitle_frame, alpha = self.create_subtitle_frame(
                        subtitle['text'], width, height, settings
                    )
                    
                    # Apply fade effect
                    if settings.get('fade_duration', 0) > 0:
                        alpha = self.apply_fade_effect(
                            alpha, current_time, subtitle['start'], 
                            subtitle['end'], settings['fade_duration']
                        )
                    
                    # Blend with main frame
                    for c in range(3):
                        frame[:, :, c] = (
                            frame[:, :, c] * (1 - alpha) + 
                            subtitle_frame[:, :, c] * alpha
                        ).astype(np.uint8)
                
                out.write(frame)
                frame_idx += 1
                
                # Update progress
                progress = frame_idx / frame_count
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_idx}/{frame_count}")
            
            # Cleanup
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Subtitles added successfully! Output: {output_path}")
            return True
            
        except Exception as e:
            st.error(f"Error adding subtitles to video: {str(e)}")
            return False
    
    def export_subtitles_srt(self, subtitles: List[Dict], output_path: str) -> bool:
        """Export subtitles to SRT format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for subtitle in subtitles:
                    # Format time
                    start_time = self.seconds_to_srt_time(subtitle['start'])
                    end_time = self.seconds_to_srt_time(subtitle['end'])
                    
                    # Write subtitle block
                    f.write(f"{subtitle['id']}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{subtitle['text']}\n\n")
            
            return True
            
        except Exception as e:
            st.error(f"Error exporting SRT file: {str(e)}")
            return False
    
    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def create_subtitle_interface(self, video_path: str, output_dir: str) -> Optional[str]:
        """Create Streamlit interface for subtitle management"""
        st.subheader("📝 Subtitle Management")
        
        # Subtitle input method
        input_method = st.radio(
            "Choose subtitle input method:",
            ["Auto-generate from text", "Upload SRT file", "Manual timing"],
            horizontal=True
        )
        
        subtitles = []
        video_duration = self.get_video_duration(video_path)
        
        if input_method == "Auto-generate from text":
            st.write("### Enter Text for Auto-Generated Subtitles")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                subtitle_text = st.text_area(
                    "Enter the text for subtitles:",
                    height=150,
                    key="auto_subtitle_text"
                )
            
            with col2:
                words_per_second = st.slider(
                    "Words per second",
                    min_value=1.0,
                    max_value=4.0,
                    value=2.5,
                    step=0.1
                )
            
            if subtitle_text:
                subtitles = self.generate_auto_subtitles(
                    subtitle_text, video_duration, words_per_second
                )
        
        elif input_method == "Upload SRT file":
            st.write("### Upload SRT File")
            srt_file = st.file_uploader(
                "Choose SRT file",
                type=['srt'],
                key="srt_upload"
            )
            
            if srt_file:
                srt_content = srt_file.read().decode('utf-8')
                subtitles = self.parse_srt_format(srt_content)
        
        elif input_method == "Manual timing":
            st.write("### Manual Subtitle Creation")
            
            # Initialize session state for manual subtitles
            if 'manual_subtitles' not in st.session_state:
                st.session_state.manual_subtitles = []
            
            # Add new subtitle
            with st.expander("Add New Subtitle"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    start_time = st.number_input(
                        "Start time (seconds)",
                        min_value=0.0,
                        max_value=video_duration,
                        value=0.0,
                        step=0.1
                    )
                
                with col2:
                    end_time = st.number_input(
                        "End time (seconds)",
                        min_value=start_time,
                        max_value=video_duration,
                        value=min(start_time + 3.0, video_duration),
                        step=0.1
                    )
                
                with col3:
                    if st.button("Add Subtitle"):
                        new_subtitle = {
                            'id': len(st.session_state.manual_subtitles) + 1,
                            'start': start_time,
                            'end': end_time,
                            'text': "",
                            'duration': end_time - start_time
                        }
                        st.session_state.manual_subtitles.append(new_subtitle)
                        st.rerun()
                
                subtitle_text = st.text_area("Subtitle text:", key="manual_text")
                
                if st.session_state.manual_subtitles and subtitle_text:
                    st.session_state.manual_subtitles[-1]['text'] = subtitle_text
            
            # Display current subtitles
            if st.session_state.manual_subtitles:
                st.write("### Current Subtitles")
                for i, sub in enumerate(st.session_state.manual_subtitles):
                    with st.expander(f"Subtitle {i+1}: {sub['start']:.1f}s - {sub['end']:.1f}s"):
                        st.write(f"**Text:** {sub['text']}")
                        if st.button(f"Remove", key=f"remove_{i}"):
                            st.session_state.manual_subtitles.pop(i)
                            st.rerun()
                
                subtitles = st.session_state.manual_subtitles
        
        # Subtitle styling options
        if subtitles:
            st.write("### Subtitle Styling")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                font_size = st.selectbox(
                    "Font Size",
                    options=list(self.font_sizes.keys()),
                    index=1
                )
            
            with col2:
                text_color = st.selectbox(
                    "Text Color",
                    options=list(self.colors.keys()),
                    index=0
                )
            
            with col3:
                bg_color = st.selectbox(
                    "Background",
                    options=list(self.bg_colors.keys()),
                    index=1
                )
            
            with col4:
                position = st.selectbox(
                    "Position",
                    options=list(self.positions.keys()),
                    index=0
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    fade_duration = st.slider(
                        "Fade Duration (seconds)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.1
                    )
                
                with col2:
                    margin = st.slider(
                        "Margin (pixels)",
                        min_value=10,
                        max_value=100,
                        value=50,
                        step=5
                    )
            
            # Create settings dictionary
            settings = {
                'font_size': font_size,
                'text_color': text_color,
                'bg_color': bg_color,
                'position': position,
                'fade_duration': fade_duration,
                'margin': margin,
                'max_chars_per_line': 50,
                'max_lines': 2
            }
            
            # Preview subtitles
            st.write("### Subtitle Preview")
            st.write(f"Total subtitles: {len(subtitles)}")
            
            if st.button("🎬 Apply Subtitles to Video"):
                output_path = os.path.join(output_dir, "video_with_subtitles.mp4")
                
                with st.spinner("Adding subtitles to video..."):
                    success = self.add_subtitles_to_video(
                        video_path, subtitles, output_path, settings
                    )
                    
                    if success:
                        # Export SRT file
                        srt_path = os.path.join(output_dir, "subtitles.srt")
                        self.export_subtitles_srt(subtitles, srt_path)
                        
                        st.success("Subtitles added successfully!")
                        
                        # Provide download links
                        col1, col2 = st.columns(2)
                        with col1:
                            if os.path.exists(output_path):
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "📥 Download Video with Subtitles",
                                        f.read(),
                                        file_name="video_with_subtitles.mp4",
                                        mime="video/mp4"
                                    )
                        
                        with col2:
                            if os.path.exists(srt_path):
                                with open(srt_path, "r", encoding="utf-8") as f:
                                    st.download_button(
                                        "📄 Download SRT File",
                                        f.read(),
                                        file_name="subtitles.srt",
                                        mime="text/plain"
                                    )
                        
                        return output_path
                    else:
                        st.error("Failed to add subtitles to video")
                        return None
        
        return None
    
    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            return duration
        except:
            return 0.0
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"Warning: Could not clean up temporary files: {str(e)}")

# Helper functions
def validate_subtitle_timing(subtitles: List[Dict]) -> List[str]:
    """Validate subtitle timing and return list of issues"""
    issues = []
    
    for i, subtitle in enumerate(subtitles):
        # Check if end time is after start time
        if subtitle['end'] <= subtitle['start']:
            issues.append(f"Subtitle {i+1}: End time must be after start time")
        
        # Check for reasonable duration
        if subtitle['duration'] < 0.5:
            issues.append(f"Subtitle {i+1}: Duration too short (< 0.5s)")
        
        if subtitle['duration'] > 10.0:
            issues.append(f"Subtitle {i+1}: Duration too long (> 10s)")
        
        # Check for overlapping subtitles
        for j, other_subtitle in enumerate(subtitles[i+1:], i+1):
            if (subtitle['start'] < other_subtitle['end'] and 
                subtitle['end'] > other_subtitle['start']):
                issues.append(f"Subtitle {i+1} overlaps with subtitle {j+1}")
    
    return issues