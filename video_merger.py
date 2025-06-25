# video_merger.py
import cv2
import os
from pathlib import Path
from typing import List, Optional
import logging
import time

if not logging.getLogger('video_merger').handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('video_merger')

class VideoMerger:
    def __init__(self):
        self.output_dir = Path("merged_videos")
        self.output_dir.mkdir(exist_ok=True)
    
    def merge_videos(self, video_paths: List[str], output_name: str = "merged_video") -> str:
        logger.info(f"Starting merge of {len(video_paths)} videos")
        valid_videos = self._validate_videos(video_paths)
        if not valid_videos: raise ValueError("No valid videos to merge.")
        
        output_filename = f"{output_name}_{int(time.time())}.mp4"
        output_path = self.output_dir / output_filename
        
        writer = None
        try:
            cap_first = cv2.VideoCapture(valid_videos[0])
            w, h, fps = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap_first.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 24
            cap_first.release()
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for i, clip_path in enumerate(valid_videos):
                logger.info(f"Processing video {i+1}/{len(valid_videos)}: {os.path.basename(clip_path)}")
                cap = cv2.VideoCapture(clip_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h))
                    writer.write(frame)
                cap.release()
        finally:
            if writer is not None: writer.release()
        logger.info(f"Videos merged successfully: {output_path}")
        return str(output_path)
    
    def add_end_card(self, video_path: str, image_path: str, duration_secs: int = 3) -> Optional[str]:
        if not os.path.exists(video_path) or not os.path.exists(image_path):
            logger.error("Video or image path not found for end card."); return None
        logger.info(f"Adding end card from '{os.path.basename(image_path)}' to video.")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_with_endcard.mp4")
        writer = None
        try:
            cap = cv2.VideoCapture(video_path)
            w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 24
            end_card_img = cv2.imread(image_path)
            end_card_img = cv2.resize(end_card_img, (w, h))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            while True:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            for _ in range(int(fps * duration_secs)):
                writer.write(end_card_img)
            cap.release()
            logger.info(f"End card added successfully to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to add end card: {e}"); return None
        finally:
            if writer is not None: writer.release()
    
    def _validate_videos(self, video_paths: List[str]) -> List[str]:
        valid_videos = []
        for path in video_paths:
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                cap = cv2.VideoCapture(path)
                if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                    valid_videos.append(path)
                cap.release()
        return valid_videos