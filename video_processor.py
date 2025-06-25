# video_processor.py
import os
import shutil
import time
import subprocess
import sys
import re
import logging
from config import (
    PYTHON_FOR_SUBPROCESS, PATH_TO_WAN21_CODE, PATH_TO_WAN21_T2V_CKPT_ROOT,
    TARGET_SCENE_DURATION_SECS, VIDEO_FPS
)

logger = logging.getLogger('video_pipeline_logger')
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(filename)s:%(lineno)d-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class VideoProcessor:
    def __init__(self):
        self.t2v_script_path = os.path.join(PATH_TO_WAN21_CODE, "generate.py")
        self.model_ckpt_path = os.path.join(PATH_TO_WAN21_T2V_CKPT_ROOT, "Wan2.1-T2V-1.3B")
        self.output_dir = "generated_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.wan21_params = {
            "video_resolution": "832*480",
            "task_name": "t2v-1.3B",
            "offload_model": "True",
            "t5_cpu": False,
            "frame_num": str(int(TARGET_SCENE_DURATION_SECS * VIDEO_FPS)),
            "sample_steps": "30",
            "guide_scale": "6.0",
            "sample_shift": "8.0"
        }
        
        self.is_initialized = self._verify_paths()
        if self.is_initialized:
            logger.info("VideoProcessor initialized. Frame count set to %s for %s-second scenes.",
                        self.wan21_params['frame_num'], TARGET_SCENE_DURATION_SECS)

    def _verify_paths(self) -> bool:
        if not os.path.isfile(self.t2v_script_path):
            logger.error(f"T2V Script not found at: {self.t2v_script_path}")
            return False
        if not os.path.isdir(self.model_ckpt_path):
            logger.error(f"T2V Checkpoint directory not found at: {self.model_ckpt_path}")
            return False
        return self._check_subprocess_cuda()

    def _check_subprocess_cuda(self) -> bool:
        try:
            cmd = [PYTHON_FOR_SUBPROCESS, "-c", "import torch; exit(0) if torch.cuda.is_available() else exit(1)"]
            subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True)
            logger.info("Subprocess Python has access to CUDA.")
            return True
        except Exception as e:
            logger.error(f"CRITICAL - Subprocess Python does NOT have access to CUDA or failed. Error: {e}")
            return False
            
    def _sanitize_filename(self, text, max_length=50):
        text = re.sub(r'[^\w\s-]', '', str(text)).strip()
        return re.sub(r'[-\s]+', '-', text)[:max_length]

    def generate_video_for_scene(self, scene_prompt: str, scene_index: int, run_id: str):
        if not self.is_initialized:
            raise RuntimeError("VideoProcessor is not initialized. Check paths in config.py.")
            
        logger.info(f"Generating Scene {scene_index+1}: {scene_prompt[:60]}...")
        
        script_ckpt_dir_arg = os.path.relpath(self.model_ckpt_path, PATH_TO_WAN21_CODE).replace("\\", "/")
        temp_save_subfolder = os.path.join("outputs", f"temp_run_{run_id}")
        abs_temp_save_folder = os.path.join(PATH_TO_WAN21_CODE, temp_save_subfolder)
        os.makedirs(abs_temp_save_folder, exist_ok=True)
        temp_save_path_arg = os.path.join(temp_save_subfolder, f"scene_{scene_index}.mp4")

        command = [
            PYTHON_FOR_SUBPROCESS, "generate.py", "--task", self.wan21_params["task_name"],
            "--size", self.wan21_params["video_resolution"], "--ckpt_dir", script_ckpt_dir_arg,
            "--prompt", scene_prompt, "--sample_guide_scale", self.wan21_params["guide_scale"],
            "--sample_shift", self.wan21_params["sample_shift"], "--frame_num", self.wan21_params["frame_num"],
            "--sample_steps", self.wan21_params["sample_steps"], "--save_file", temp_save_path_arg,
            "--offload_model", self.wan21_params["offload_model"]
        ]
        if self.wan21_params["t5_cpu"]: command.append("--t5_cpu")

        try:
            logger.debug(f"Executing command: {' '.join(command)}")
            # Increased timeout to 30 minutes to be safe for complex scenes
            result = subprocess.run(
                command, 
                cwd=PATH_TO_WAN21_CODE, 
                capture_output=True, 
                text=True, 
                timeout=60 * 30,  # 30-minute timeout
                check=True
            )
            
            temp_file_path = os.path.join(PATH_TO_WAN21_CODE, temp_save_path_arg)
            
            # --- START OF THE FIX ---
            # Robustness Check 1: Verify the file was actually created.
            if not os.path.exists(temp_file_path):
                logger.error(f"CRITICAL: 'generate.py' finished but did not create the output file at {temp_file_path}")
                logger.error(f"'generate.py' STDOUT:\n{result.stdout}")
                logger.error(f"'generate.py' STDERR:\n{result.stderr}")
                raise RuntimeError(f"Video file for Scene {scene_index+1} was not generated by the subprocess.")
            
            # Robustness Check 2: Verify the file is not empty.
            if os.path.getsize(temp_file_path) < 1024: # Check if file is smaller than 1KB
                 logger.error(f"CRITICAL: Video file for Scene {scene_index+1} was generated but is empty or corrupt.")
                 logger.error(f"'generate.py' STDOUT:\n{result.stdout}")
                 logger.error(f"'generate.py' STDERR:\n{result.stderr}")
                 raise RuntimeError(f"Video file for Scene {scene_index+1} appears to be empty or corrupt.")
            # --- END OF THE FIX ---

            final_filename = f"scene_{scene_index+1}_{self._sanitize_filename(scene_prompt)}.mp4"
            final_dest_path = os.path.join(self.output_dir, final_filename)
            shutil.move(temp_file_path, final_dest_path)
            
            logger.info(f"Scene {scene_index+1} video saved to: {final_dest_path}")
            return final_dest_path
            
        except subprocess.CalledProcessError as e:
            # This catches cases where generate.py returns a non-zero exit code (an explicit crash).
            logger.error(f"'generate.py' process failed for Scene {scene_index+1} with exit code {e.returncode}.")
            logger.error(f"Stderr from 'generate.py':\n{e.stderr}")
            logger.error(f"Stdout from 'generate.py':\n{e.stdout}")
            raise RuntimeError(f"'generate.py' process failed. See server logs for detailed stderr.")
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"'generate.py' timed out for Scene {scene_index+1} after {e.timeout} seconds.")
            logger.error(f"Stderr so far:\n{e.stderr}")
            raise RuntimeError(f"'generate.py' process timed out.")
            
        except FileNotFoundError as e:
            # This explicitly catches the case where shutil.move fails because the source file is missing.
            logger.error(f"Failed to move video file because it was not found at the expected path: {e.filename}", exc_info=True)
            raise RuntimeError(f"Could not find the generated video file for Scene {scene_index+1}. The generation subprocess may have failed silently.")

        except Exception as e:
            # A general catch-all for other unexpected errors.
            logger.error(f"An unexpected error occurred in generate_video_for_scene: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred while processing Scene {scene_index+1}: {e}")