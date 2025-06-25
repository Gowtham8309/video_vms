# prompt_restructurer.py
import logging
from typing import Optional
from scene_generator import SceneGenerator # We can reuse the same Mistral model loader

logger = logging.getLogger('prompt_restructurer_logger')
if not logger.handlers:
    # Basic logger setup
    pass

class PromptRestructurer:
    def __init__(self, scene_generator: SceneGenerator):
        """
        Initializes the restructurer. It shares the same Mistral model
        as the SceneGenerator to conserve memory.
        """
        self.scene_generator = scene_generator
        logger.info("PromptRestructurer initialized, sharing Mistral model.")

    def _create_restructure_prompt(self, original_prompt: str, iteration_num: int, category: str) -> str:
        """
        Creates a sophisticated prompt to guide Mistral in restructuring the user's prompt
        while preserving the core theme.
        """
        # Iteration-specific instructions to guide the evolution
        focus_prompts = {
            2: "Introduce a new angle or perspective. Focus on a different detail or emotion.",
            3: "Expand the scope. Think about the 'before' or 'after' of the original scene.",
            4: "Increase the intensity or dynamism. Add more action or sensory details.",
            5: "Create a concluding or summary version. Bring the theme to a powerful close."
        }
        
        focus = focus_prompts.get(iteration_num, "Refine and enhance the original concept.")

        return f"""
[INST]
You are an expert creative director. Your task is to take a core video concept and subtly evolve it for the next iteration of a video ad campaign.

**Rules:**
1.  **Preserve the Core Theme:** The restructured prompt MUST be about the same subject and maintain the original intent. Do NOT change the product, service, or central idea.
2.  **Evolve, Don't Replace:** Your goal is to offer a fresh take, not a completely new idea.
3.  **Follow the Iteration Focus:** Use the specific focus for this iteration to guide your creative changes.
4.  **Output Only the Prompt:** Provide only the new, restructured prompt text. Do not add any commentary.

**Original Core Prompt:** "{original_prompt}"
**Business Category:** "{category}"
**Current Iteration:** {iteration_num}/5
**This Iteration's Creative Focus:** {focus}

Restructured Prompt:
[/INST]
"""

    def restructure_prompt(self, original_prompt: str, category: str, iteration_count: int) -> str:
        """
        Restructures the prompt for the given iteration number (1-based).
        """
        if not self.scene_generator.is_initialized:
            logger.warning("Mistral model not ready. Returning original prompt.")
            return f"{original_prompt} (iteration {iteration_count + 1})"

        iteration_num = iteration_count + 1 # Convert 0-based index to 1-based number
        if iteration_num <= 1:
            return original_prompt # The first iteration always uses the original prompt

        logger.info(f"Restructuring prompt for iteration {iteration_num}...")
        
        full_prompt = self._create_restructure_prompt(original_prompt, iteration_num, category)
        model = self.scene_generator.model
        tokenizer = self.scene_generator.tokenizer

        model_inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.75,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        restructured_prompt = response_text.split("[/INST]")[-1].strip().strip('"')

        # Fallback in case Mistral gives an empty response
        if not restructured_prompt:
            logger.warning("Prompt restructuring returned empty. Using original prompt.")
            return original_prompt

        logger.info(f"New prompt for Iteration {iteration_num}: {restructured_prompt}")
        return restructured_prompt