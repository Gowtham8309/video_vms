# main.py
import streamlit as st
import os
import time
import cv2
import tempfile
from config import NUM_SCENES_PER_ITERATION, BRAND_ASSETS_PATH
from video_processor import VideoProcessor
from scene_generator import SceneGenerator
from video_merger import VideoMerger
from audio_manager import AudioManager
from prompt_restructurer import PromptRestructurer
from history_manager import HistoryManager

st.set_page_config(page_title="VMS", page_icon="🎬", layout="wide")

@st.cache_resource
def initialize_managers():
    scene_gen = SceneGenerator()
    return {
        'video_processor': VideoProcessor(), 'scene_generator': scene_gen,
        'prompt_restructurer': PromptRestructurer(scene_gen), 'video_merger': VideoMerger(),
        'audio_manager': AudioManager(), 'history_manager': HistoryManager()
    }

# --- Session State Initialization ---
default_state = {
    'page': 'New Generation', 'processing_stage': 'input', 'iteration_count': 0,
    'final_videos': [], 'post_processing_choice': None, 'is_playing': False,
    'locked_audio_path': None, 'locked_image_path': None, 'current_prompt': "",
    'original_prompt': "", 'current_video_index': 0, 'player_id': None
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

managers = initialize_managers()

def get_video_duration(video_path: str) -> float:
    """Safely get the duration of a video file in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps and fps > 0 else 10.0 # Default to 10s if error
        cap.release()
        return duration
    except Exception:
        return 10.0

def main():
    st.sidebar.title("Navigation")
    def page_change_callback():
        st.session_state.is_playing = False
        st.session_state.current_video_index = 0
    st.sidebar.radio("Go to", ["New Generation", "Past Generations"], key="page", on_change=page_change_callback)
    
    if st.session_state.page == "New Generation":
        run_main_workflow()
    else:
        show_history_page()

def run_main_workflow():
    st.title("🎬 VMS - New Generation")
    with st.sidebar:
        st.header("📊 Process Tracking"); st.metric("Current Iteration", f"{st.session_state.iteration_count + 1}/5")
        st.header("⚙️ Settings"); st.info(f"Scenes per Iteration: {NUM_SCENES_PER_ITERATION}")
        if st.session_state.post_processing_choice: st.info(f"Post-Processing: {st.session_state.post_processing_choice.replace('_', ' ').title()}")
        st.header("🎯 Current Stage"); st.info(f"Stage: {st.session_state.processing_stage.title()}")

    if st.session_state.processing_stage == "input": show_input_section()
    elif st.session_state.processing_stage == "generation": show_generation_section()
    elif st.session_state.processing_stage == "post_processing": show_post_processing_section()
    elif st.session_state.processing_stage == "completed": show_completion_section(st.session_state.final_videos)

def show_input_section():
    st.header("📝 Input Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Video Prompt"); prompt_input = st.text_area("Enter video prompt:", height=150, key="prompt_input")
        uploaded_image = st.file_uploader("Upload optional brand image (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded_image:
            temp_dir = tempfile.gettempdir(); image_path = os.path.join(temp_dir, uploaded_image.name)
            with open(image_path, "wb") as f: f.write(uploaded_image.getbuffer())
            st.session_state.locked_image_path = image_path; st.success(f"Image '{uploaded_image.name}' uploaded.")
    with col2:
        st.subheader("🏢 Business Category")
        categories = ["🤖 SensesMind 360", "🏨 Hotels & Resorts", "💍 Jewelry Brands", "👗 Fashion Malls", "🛒 Retail Chains", "🍽️ Restaurants", "🏥 Healthcare & Clinics", "🎓 Education Institutes", "🏢 Corporate Business", "🎪 Entertainment", "🏃 Sports & Fitness"]
        selected_category = st.selectbox("Choose business category:", categories, key="category_select")
    if st.button("🚀 Generate Videos", type="primary", use_container_width=True):
        if prompt_input and prompt_input.strip():
            for key in ['iteration_count','final_videos','post_processing_choice','locked_audio_path','is_playing','current_video_index','player_id']:
                st.session_state[key] = 0 if key in ['iteration_count', 'current_video_index'] else [] if key == 'final_videos' else False if key == 'is_playing' else None
            st.session_state.original_prompt, st.session_state.current_prompt = prompt_input, prompt_input
            st.session_state.selected_category = selected_category; st.session_state.processing_stage = "generation"; st.rerun()
        else: st.error("Please provide a text prompt!")
    
def auto_post_process(video_path, current_prompt):
    if st.session_state.selected_category == "🤖 SensesMind 360":
        st.info("SensesMind 360 category: Skipping audio/subtitle post-processing."); return video_path
    choice = st.session_state.post_processing_choice
    if choice is None or choice == 'skip': return video_path
    st.info(f"Automatically applying '{choice.replace('_', ' ')}' post-processing...")
    am, sg = managers['audio_manager'], managers['scene_generator']
    output_dir = "final_videos"; os.makedirs(output_dir, exist_ok=True)
    try:
        final_path = None
        if choice == 'audio': final_path = am.add_department_audio(video_path, output_dir, st.session_state.selected_category, st.session_state.locked_audio_path)
        elif choice == 'subtitles': final_path = am.add_summarized_subtitles(video_path, output_dir, current_prompt, sg)
        elif choice == 'both': final_path = am.add_audio_and_subtitles(video_path, output_dir, st.session_state.selected_category, current_prompt, sg, st.session_state.locked_audio_path)
        return final_path if final_path and os.path.exists(final_path) else video_path
    except Exception as e:
        st.error(f"Auto post-processing failed: {e}"); return video_path

def show_generation_section():
    st.header(f"🎬 Iteration {st.session_state.iteration_count + 1}: Generation")
    if st.session_state.iteration_count > 0:
        with st.spinner("Evolving prompt..."):
            st.session_state.current_prompt = managers['prompt_restructurer'].restructure_prompt(st.session_state.original_prompt, st.session_state.selected_category, st.session_state.iteration_count)
    st.info(f"**Current Prompt:** {st.session_state.current_prompt}")
    progress_bar = st.progress(0, text="Initializing...")
    try:
        scenes = managers['scene_generator'].generate_scenes(st.session_state.current_prompt, st.session_state.selected_category)
        scene_videos, run_id = [], f"run_{int(time.time())}"
        for i, scene in enumerate(scenes):
            progress_bar.progress(10 + int((i / len(scenes)) * 65), text=f"Generating scene {i+1}/{len(scenes)}...")
            video_path = managers['video_processor'].generate_video_for_scene(scene, i, run_id)
            scene_videos.append(video_path)
        progress_bar.progress(80, text="Merging clips...")
        merged_video_path = managers['video_merger'].merge_videos(scene_videos, f"iteration_{st.session_state.iteration_count + 1}")
        image_for_end_card = st.session_state.locked_image_path
        if not image_for_end_card and st.session_state.selected_category == "🤖 SensesMind 360":
            logo_path = os.path.join(BRAND_ASSETS_PATH, "sensesmind_logo.png")
            if os.path.exists(logo_path): image_for_end_card = logo_path
        if image_for_end_card:
            progress_bar.progress(90, text="Adding end card...")
            video_with_endcard = managers['video_merger'].add_end_card(merged_video_path, image_for_end_card)
            if video_with_endcard: merged_video_path = video_with_endcard
        if st.session_state.post_processing_choice:
            final_processed_path = auto_post_process(merged_video_path, st.session_state.current_prompt)
            st.session_state.final_videos.append(final_processed_path)
            proceed_to_next_iteration()
        else:
            st.session_state.unprocessed_video_path = merged_video_path
            st.session_state.processing_stage = "post_processing"
        st.success("Iteration generation complete!"); time.sleep(1); st.rerun()
    except Exception as e: st.error(f"Error during generation: {e}")

def show_post_processing_section():
    st.header("🎵 Select Post-Processing (This choice will be used for all 5 iterations)")
    video_path = st.session_state.get('unprocessed_video_path')
    if not video_path or not os.path.exists(video_path):
        st.error("No video for post-processing. Restarting."); st.session_state.clear(); st.rerun(); return
    st.video(video_path)
    if st.session_state.selected_category == "🤖 SensesMind 360":
        st.warning("For 'SensesMind 360', custom audio and subtitles are disabled.")
        if st.button("Continue", use_container_width=True):
            st.session_state.post_processing_choice = 'skip'
            st.session_state.final_videos.append(video_path)
            proceed_to_next_iteration()
        return
    am, sg = managers['audio_manager'], managers['scene_generator']
    output_dir = "final_videos"; os.makedirs(output_dir, exist_ok=True)
    col1, col2, col3, col4 = st.columns(4)
    choice = None
    if col1.button("🎵 Add Audio Only", use_container_width=True): choice = 'audio'
    if col2.button("📝 Add Subtitles Only", use_container_width=True): choice = 'subtitles'
    if col3.button("🎵📝 Add Both", use_container_width=True): choice = 'both'
    if col4.button("⏩ Skip for all", use_container_width=True): choice = 'skip'
    if choice:
        st.session_state.post_processing_choice = choice
        with st.spinner(f"Applying '{choice}' post-processing..."):
            final_path = am.add_department_audio(video_path, output_dir, st.session_state.selected_category) if choice == 'audio' else \
                         am.add_summarized_subtitles(video_path, output_dir, st.session_state.original_prompt, sg) if choice == 'subtitles' else \
                         am.add_audio_and_subtitles(video_path, output_dir, st.session_state.selected_category, st.session_state.original_prompt, sg) if choice == 'both' else \
                         video_path
        st.session_state.final_videos.append(final_path if final_path else video_path)
        proceed_to_next_iteration()

def proceed_to_next_iteration():
    st.session_state.iteration_count += 1
    if st.session_state.iteration_count == 2:
        st.session_state.processing_stage = "completed"
        managers['history_manager'].add_generation_to_history(st.session_state.original_prompt, st.session_state.selected_category, st.session_state.final_videos)
    else:
        st.session_state.processing_stage = "generation"
    st.rerun()

def show_completion_section(video_playlist):
    st.header("🎉 All Iterations Completed!")
    st.success("Your final videos are ready for playback.")
    if not video_playlist:
        st.warning("No final videos were found."); return

    st.subheader("Continuous Video Playback")
    
    playlist_id = video_playlist[0] if video_playlist else "default"
    
    col1, col2 = st.columns([1, 4])
    if col1.button("▶️ Play All", use_container_width=True, key=f"play_{playlist_id}"):
        st.session_state.is_playing = True
        st.session_state.current_video_index = 0
        st.session_state.player_id = playlist_id
        st.rerun()

    if col1.button("⏹️ Stop", use_container_width=True, key=f"stop_{playlist_id}"):
        st.session_state.is_playing = False
        st.rerun()
    
    video_placeholder = st.empty()

    # --- THIS IS THE CORRECTED AUTOPLAY LOGIC ---
    if not st.session_state.get('is_playing', False) or st.session_state.get('player_id') != playlist_id:
        with video_placeholder.container():
            st.info("Playback is stopped. Press 'Play All' to begin.")
            current_index = st.session_state.get('current_video_index', 0)
            if video_playlist and current_index < len(video_playlist): 
                st.video(video_playlist[current_index])
    else:
        current_index = st.session_state.get('current_video_index', 0)
        
        while st.session_state.get('is_playing', False):
            # Loop back to the beginning if we've finished the playlist
            if current_index >= len(video_playlist):
                current_index = 0
            
            video_path_to_play = video_playlist[current_index]
            
            with video_placeholder.container():
                st.info(f"Now Playing: Iteration {current_index + 1}/{len(video_playlist)}")
                st.warning("Player is in automated mode. 'Stop' will take effect after this video finishes.")
                st.video(video_path_to_play)
            
            # Get the ACTUAL duration of the current video
            duration = get_video_duration(video_path_to_play)
            
            # This will freeze the app for the duration of the video
            time.sleep(duration)
            
            # Prepare for the next loop
            current_index += 1
            st.session_state.current_video_index = current_index
            
            # Rerun the script to check the 'is_playing' flag and display the next video
            st.rerun()

    if st.button("🔄 Start New Project", key=f"restart_{playlist_id}"):
        keys_to_keep = ['_managers', 'page']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep: del st.session_state[key]
        st.rerun()

def show_history_page():
    st.title("📖 Past Generations History")
    all_sessions = managers['history_manager'].get_all_sessions()
    if not all_sessions:
        st.info("No past generations found."); return
        
    session_options = {f"{s['date']} - {s['original_prompt'][:40]}...": s['id'] for s in all_sessions}
    selected_display = st.selectbox("Select a past session to view:", options=session_options.keys())

    if selected_display:
        session_id = session_options[selected_display]
        session_data = managers['history_manager'].get_session_by_id(session_id)
        st.subheader(f"Details for session from {session_data['date']}")
        st.text(f"Prompt: {session_data['original_prompt']}"); st.text(f"Category: {session_data['category']}")
        if 'last_session_id' not in st.session_state or st.session_state.last_session_id != session_id:
            st.session_state.current_video_index = 0; st.session_state.is_playing = False; st.session_state.last_session_id = session_id
        show_completion_section(session_data['final_videos'])

if __name__ == "__main__":
    main()