# config.py
import os
import sys

PYTHON_FOR_SUBPROCESS = sys.executable
PATH_TO_WAN21_CODE = "/root/Vms/Wan2.1_code"
PATH_TO_WAN21_T2V_CKPT_ROOT = "/root/Vms/models"
DEPARTMENT_AUDIO_ROOT_PATH = "department_audio"
BRAND_ASSETS_PATH = "brand_assets"

# --- GENERATION SETTINGS ---
NUM_SCENES_PER_ITERATION = 2
TARGET_SCENE_DURATION_SECS = 2
VIDEO_FPS = 24

# --- Sanity Checks ---
if not os.path.exists(PYTHON_FOR_SUBPROCESS): raise FileNotFoundError(f"Python executable not found at: {PYTHON_FOR_SUBPROCESS}")
if not os.path.isdir(PATH_TO_WAN21_CODE): raise FileNotFoundError(f"Wan2.1 code directory not found at: {PATH_TO_WAN21_CODE}")
if not os.path.isdir(PATH_TO_WAN21_T2V_CKPT_ROOT): raise FileNotFoundError(f"Model checkpoint root directory not found at: {PATH_TO_WAN21_T2V_CKPT_ROOT}")
if not os.path.isdir(DEPARTMENT_AUDIO_ROOT_PATH): raise FileNotFoundError(f"Department audio directory not found at: {DEPARTMENT_AUDIO_ROOT_PATH}")
if not os.path.isdir(BRAND_ASSETS_PATH): raise FileNotFoundError(f"Brand assets directory not found at: {BRAND_ASSETS_PATH}")