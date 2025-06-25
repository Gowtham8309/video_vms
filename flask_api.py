# flask_api.py
import os
import time
import logging
from flask import Flask, request, jsonify, abort, send_from_directory

# --- Import all your project modules (no changes needed here) ---
from config import BRAND_ASSETS_PATH, NUM_SCENES_PER_ITERATION
from video_processor import VideoProcessor
from scene_generator import SceneGenerator
from video_merger import VideoMerger
from audio_manager import AudioManager
from prompt_restructurer import PromptRestructurer
from history_manager import HistoryManager

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Models and Managers ONCE on startup ---
# This logic is identical to the FastAPI version.
logger.info("Initializing all managers for Flask. This may take a moment...")
try:
    scene_gen = SceneGenerator()
    MANAGERS = {
        'video_processor': VideoProcessor(),
        'scene_generator': scene_gen,
        'prompt_restructurer': PromptRestructurer(scene_gen),
        'video_merger': VideoMerger(),
        'audio_manager': AudioManager(),
        'history_manager': HistoryManager()
    }
    logger.info("All managers initialized successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize managers. API cannot start. Error: {e}", exc_info=True)
    MANAGERS = None

# --- Main Generation Endpoint ---

@app.route("/generate", methods=['POST'])
def generate_videos():
    """
    Generate a series of videos based on a prompt and other parameters.
    """
    if not MANAGERS:
        # Use jsonify for a proper JSON error response
        return jsonify(error="Service Unavailable: Managers are not initialized."), 503

    # --- Manual Request Parsing and Validation (Flask's way) ---
    data = request.get_json()
    if not data:
        return jsonify(error="Invalid request: JSON body required."), 400

    prompt = data.get('prompt')
    category = data.get('category')
    
    if not prompt or not category:
        return jsonify(error="Missing required fields: 'prompt' and 'category' are required."), 400
        
    post_processing_choice = data.get('post_processing_choice', 'skip')
    num_iterations = data.get('num_iterations', 2)
    end_card_image_name = data.get('end_card_image_name')
    # --- End of manual validation ---

    run_id = f"flask_run_{int(time.time())}"
    logger.info(f"Starting new generation run '{run_id}' for prompt: '{prompt[:50]}...'")

    final_video_paths = []
    original_prompt = prompt
    current_prompt = original_prompt

    try:
        for i in range(num_iterations):
            iteration_num = i + 1
            logger.info(f"[{run_id}] Starting Iteration {iteration_num}/{num_iterations}")

            # 1. Evolve prompt (identical logic)
            if i > 0:
                logger.info(f"[{run_id}] Evolving prompt...")
                current_prompt = MANAGERS['prompt_restructurer'].restructure_prompt(
                    original_prompt, category, i
                )
            logger.info(f"[{run_id}] Current prompt: {current_prompt}")

            # 2. Generate scenes (identical logic)
            scenes = MANAGERS['scene_generator'].generate_scenes(current_prompt, category)
            
            # 3. Generate video clips (identical logic)
            scene_videos = []
            for scene_index, scene_prompt in enumerate(scenes):
                logger.info(f"[{run_id}] Generating video for scene {scene_index+1}: '{scene_prompt[:40]}...'")
                video_path = MANAGERS['video_processor'].generate_video_for_scene(
                    scene_prompt, scene_index, run_id
                )
                scene_videos.append(video_path)

            # 4. Merge clips (identical logic)
            logger.info(f"[{run_id}] Merging {len(scene_videos)} clips...")
            merged_video_path = MANAGERS['video_merger'].merge_videos(
                scene_videos, f"iteration_{iteration_num}_{run_id}"
            )

            # 5. Add end card (identical logic)
            video_with_endcard = merged_video_path
            if end_card_image_name:
                image_path = os.path.join(BRAND_ASSETS_PATH, end_card_image_name)
                if os.path.exists(image_path):
                    logger.info(f"[{run_id}] Adding end card from '{end_card_image_name}'...")
                    video_with_endcard = MANAGERS['video_merger'].add_end_card(merged_video_path, image_path) or merged_video_path
                else:
                    logger.warning(f"[{run_id}] End card image not found: {image_path}. Skipping.")
            
            # 6. Post-processing (identical logic)
            final_path = video_with_endcard
            output_dir = "final_videos"
            
            if post_processing_choice != 'skip':
                logger.info(f"[{run_id}] Applying post-processing: {post_processing_choice}")
                am = MANAGERS['audio_manager']
                sg = MANAGERS['scene_generator']
                
                if post_processing_choice == 'audio':
                    final_path = am.add_department_audio(video_with_endcard, output_dir, category)
                elif post_processing_choice == 'subtitles':
                    final_path = am.add_summarized_subtitles(video_with_endcard, output_dir, current_prompt, sg)
                elif post_processing_choice == 'both':
                    final_path = am.add_audio_and_subtitles(video_with_endcard, output_dir, category, current_prompt, sg)
            
            final_video_paths.append(final_path or video_with_endcard)
            logger.info(f"[{run_id}] Iteration {iteration_num} complete. Final video at: {final_path}")

        # 7. Save to history (identical logic)
        session_data = MANAGERS['history_manager'].add_generation_to_history(
            original_prompt, category, final_video_paths
        )
        session_id = session_data.get("id", run_id)

        # 8. Create full URLs for the response (Flask's way)
        base_url = request.host_url
        final_video_urls = []
        for path in final_video_paths:
            filename = os.path.basename(path)
            if path.startswith("final_videos"):
                final_video_urls.append(f"{base_url}final_videos/{filename}")
            elif path.startswith("merged_videos"):
                 final_video_urls.append(f"{base_url}merged_videos/{filename}")

        logger.info(f"[{run_id}] Generation complete. Returning {len(final_video_urls)} video URLs.")
        
        return jsonify({
            "session_id": session_id,
            "original_prompt": original_prompt,
            "final_video_urls": final_video_urls,
            "message": "Video generation process completed successfully."
        })

    except Exception as e:
        logger.error(f"[{run_id}] An error occurred during video generation: {e}", exc_info=True)
        return jsonify(error=f"An internal error occurred: {str(e)}"), 500


# --- Routes for Serving Generated Files ---
# Flask requires explicit routes to serve files from directories.

@app.route('/merged_videos/<path:filename>')
def serve_merged_video(filename):
    return send_from_directory('merged_videos', filename)

@app.route('/final_videos/<path:filename>')
def serve_final_video(filename):
    # Ensure the final_videos directory exists
    os.makedirs("final_videos", exist_ok=True)
    return send_from_directory('final_videos', filename)

@app.route('/assets/<path:filename>')
def serve_brand_asset(filename):
    return send_from_directory(BRAND_ASSETS_PATH, filename)


# --- Run the App ---
if __name__ == "__main__":
    # The debug=True flag provides auto-reloading and helpful error pages.
    # Do not use debug=True in a production environment.
    app.run(host="0.0.0.0", port=8000, debug=True)