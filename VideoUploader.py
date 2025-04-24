# from __future__ import annotations

import os
import folder_paths
import hashlib

class VideoUploader:
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        video_files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["video"])
        return {
            "required": {
                "video": (sorted(video_files), {"video_upload": True})
            }
        }

    RETURN_TYPES = ("VIDEO_PATH",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "upload_video"
    CATEGORY = "🐼QwenOmni"

    def upload_video(self, video):
        video_path = folder_paths.get_annotated_filepath(video) if video else None
        return (video_path,)

    @classmethod
    def IS_CHANGED(self, video):
        video_path = folder_paths.get_annotated_filepath(video)
        m = hashlib.sha256()
        with open(video_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(self, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True