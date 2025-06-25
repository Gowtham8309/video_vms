# history_manager.py
import json
import os
from datetime import datetime

class HistoryManager:
    def __init__(self, history_file='generation_history.json'):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self) -> list:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

    def add_generation_to_history(self, original_prompt: str, category: str, final_video_paths: list):
        session_data = {
            "id": f"session_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_prompt": original_prompt,
            "category": category,
            "final_videos": final_video_paths
        }
        self.history.insert(0, session_data)
        self._save_history()
        # --- THIS IS THE FIX ---
        return session_data
        # -----------------------

    def get_all_sessions(self) -> list:
        return self.history

    def get_session_by_id(self, session_id: str) -> dict | None:
        for session in self.history:
            if session.get('id') == session_id:
                return session
        return None