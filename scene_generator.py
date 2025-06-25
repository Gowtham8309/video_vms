# scene_generator.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import re
import os # <--- Import os
from typing import List
from config import NUM_SCENES_PER_ITERATION

# --- Add this line to get the token ---
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN", None)

logger = logging.getLogger('scene_generator_logger')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SceneGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.is_initialized = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            if not torch.cuda.is_available():
                logger.error("CUDA not available for Mistral 7B.")
                return

            if HF_TOKEN is None:
                logger.warning("Hugging Face token not found in environment variables. Set HUGGING_FACE_TOKEN.")
                # You could also choose to raise an error here
                # raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")

            logger.info(f"Initializing Scene Generator with model: {self.model_id}")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            
            # --- Update these two lines to include the token ---
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                quantization_config=quantization_config, 
                device_map="auto",
                token=HF_TOKEN
            )
            # --- End of update ---

            self.is_initialized = True
            logger.info("Mistral 7B model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Mistral 7B model: {e}", exc_info=True)

    def _create_scene_prompt_template(self, main_prompt: str, category: str, num_scenes: int) -> str:
        return f"""[INST]You are a creative assistant for a video production company. Your task is to break down a high-level video concept into a sequence of distinct, visually descriptive scenes.

**Instructions:**
1. Read the "Main Video Prompt" and "Business Category".
2. Generate exactly {num_scenes} distinct scenes.
3. Each scene description must be a single, concise line, focusing on visual elements.
4. Do not add any commentary, just the numbered list of scenes.

**Main Video Prompt:** "{main_prompt}"
**Business Category:** "{category}"
[/INST]"""

    def generate_scenes(self, main_prompt: str, category: str) -> List[str]:
        num_scenes = NUM_SCENES_PER_ITERATION
        if not self.is_initialized:
            logger.error("Scene generator model not initialized. Using fallback.")
            return [f"Scene {i+1}: {main_prompt}" for i in range(num_scenes)]
            
        logger.info(f"Generating {num_scenes} scenes for prompt: '{main_prompt[:50]}...'")
        full_prompt = self._create_scene_prompt_template(main_prompt, category, num_scenes)
        model_inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response_content = response_text.split("[/INST]")[-1].strip()
        
        scenes = re.findall(r'^\d+\.\s*(.*)', response_content, re.MULTILINE)
        if not scenes or len(scenes) < num_scenes:
            scenes = [s.strip() for s in response_content.split('\n') if s.strip()]
            scenes = [re.sub(r'^\d+\.\s*', '', s) for s in scenes if s]

        logger.info(f"Generated scenes: {scenes[:num_scenes]}")
        return scenes[:num_scenes]

    def summarize_prompt_for_subtitles(self, main_prompt: str) -> str:
        if not self.is_initialized:
            return f"A video about: {main_prompt[:50]}..."

        logger.info(f"Summarizing prompt for subtitles: '{main_prompt[:50]}...'")
        prompt = f"""[INST]You are a marketing copywriter. Summarize the following video concept into a single, concise, and impactful sentence for a video subtitle.

**Video Concept:** "{main_prompt}"

**Summarized Sentence:**
[/INST]"""
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50, do_sample=True, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id)

        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        summary = response_text.split("[/INST]")[-1].strip().strip('"')
        
        logger.info(f"Generated subtitle summary: {summary}")
        return summary