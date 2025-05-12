from __future__ import annotations
import torch
import os
import tempfile
import io
import torchaudio
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import requests
import time
from .VideoUploader import VideoUploader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU（如RTX 30系及以上）
    except ImportError:
        return False

FLASH_ATTENTION_AVAILABLE = check_flash_attention()

class QwenModelLoader:
    """处理Qwen模型加载的兼容性问题"""
    @staticmethod
    def get_model_class():
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration
            logger.info("使用原生 Qwen2_5OmniForConditionalGeneration 模型类")
            return Qwen2_5OmniForConditionalGeneration
        except ImportError:
            logger.warning("Qwen2_5OmniForConditionalGeneration 不可用，使用 AutoModelForCausalLM 替代")
            return AutoModelForCausalLM

def init_qwen_paths():
    """动态注册模型路径（支持ComfyUI模型管理）"""
    qwen_dir = Path(folder_paths.models_dir) / "Qwen"
    model_dir = qwen_dir / "Qwen2.5-Omni-7B"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 兼容ComfyUI新旧版本路径注册
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(model_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(model_dir)], {'.safetensors', '.bin'})

    return str(model_dir)

def test_download_speed(url):
    """测试下载速度，下载 5 秒"""
    try:
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=10)
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            if time.time() - start_time > 5:
                break
            downloaded_size += len(data)
        end_time = time.time()
        speed = downloaded_size / (end_time - start_time) / 1024  # KB/s
        return speed
    except Exception as e:
        logger.error(f"测试下载速度时出现错误: {e}")
        return 0

def check_model_files_exist(model_dir):
    """检查模型文件是否齐全"""
    required_files = [
        "added_tokens.json",
        "chat_template.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "spk_dict.pt",
        "tokenizer.json",
        "vocab.json",
        "config.json",
        "generation_config.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json"
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True

class QwenOmniCombined:
    def __init__(self):
        self.model_path = init_qwen_paths()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_class = QwenModelLoader.get_model_class()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["Qwen2.5-Omni-7B"],
                    {"default": "Qwen2.5-Omni-7B"}
                ),
                "quantization": (
                    ["👍 4-bit (VRAM-friendly)", "⚖️ 8-bit (Balanced Precision)", "🚫 None (Original Precision)"],
                    {"default": "👍 4-bit (VRAM-friendly)"}
                ),
                "prompt": ("STRING", {"default": "Hi!😽", "multiline": True}),
                "audio_output": (
                    ["🔇None (No Audio)", "👱‍♀️Chelsie (Female)", "👨‍🦰Ethan (Male)"],
                    {"default": "🔇None (No Audio)"}
                ),
                "audio_source": (
                    ["🎧 Separate Audio Input", "🎬 Video Built-in Audio"],
                    {"default": "🎧 Separate Audio Input", "display": "radio"}
                ),
                "max_tokens": ("INT", {"default": 132, "min": 64, "max": 2048, "step": 16, "display": "slider"}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.1, "display": "slider"}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"})
            },
            "optional": {
                "image": ("IMAGE", {}),
                "audio": ("AUDIO", {}),
                "video_path": ("VIDEO_PATH", {})
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "process"
    CATEGORY = "🐼QwenOmni"

    def load_model(self, model_name, quantization):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for {model_name} model")

        quant_config = None
        if quantization == "👍 4-bit (VRAM-friendly)":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "⚖️ 8-bit (Balanced Precision)":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        device_map = {"": 0} if torch.cuda.device_count() > 0 else "auto"

        if not check_model_files_exist(self.model_path):
            self.download_model()

        logger.info(f"使用模型类: {self.model_class.__name__}")
        self.model = self.model_class.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "sdpa",
            trust_remote_code=True
        ).eval()

        if torch.__version__ >= "2.2":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def download_model(self):
        """处理模型下载逻辑"""
        huggingface_test_url = "https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main/model-00005-of-00005.safetensors"
        modelscope_test_url = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-Omni-7B/repo?Revision=master&FilePath=model-00005-of-00005.safetensors"
        
        huggingface_speed = test_download_speed(huggingface_test_url)
        modelscope_speed = test_download_speed(modelscope_test_url)

        download_sources = [
            (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face"),
            (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope")
        ] if huggingface_speed >= modelscope_speed else [
            (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope"),
            (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face")
        ]

        max_retries = 3
        for download_func, repo_id, source in download_sources:
            for retry in range(max_retries):
                logger.info(f"尝试从 {source} 下载模型 (尝试 {retry + 1}/{max_retries})")
                try:
                    kwargs = {"cache_dir": self.model_path}
                    if download_func == snapshot_download:
                        kwargs["ignore_patterns"] = ["*.msgpack", "*.h5"]
                    download_func(repo_id, **kwargs)
                    logger.info(f"成功从 {source} 下载模型")
                    return
                except Exception as e:
                    logger.warning(f"下载失败: {e}")
                    if retry == max_retries - 1:
                        logger.error(f"从 {source} 下载失败")
                    else:
                        time.sleep(2 ** retry)  # 指数退避

        raise RuntimeError("所有下载源均失败")

    # ... [保留原有的 tensor_to_pil 和 process 方法不变]
    # 注意：这里应该保留原有的 process 方法实现，只是没有完整贴出以节省空间

NODE_CLASS_MAPPINGS = {
    "VideoUploader": VideoUploader,
    "QwenOmniCombined": QwenOmniCombined
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUploader": "Video Uploader🐼",
    "QwenOmniCombined": "Qwen Omni Combined🐼"
}
