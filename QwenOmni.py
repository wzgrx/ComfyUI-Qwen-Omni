from __future__ import annotations
import torch
import os
import tempfile
import io
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import datetime
import hashlib
import requests
import time
from .VideoUploader import VideoUploader


def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU（如RTX 30系及以上）
    except ImportError:
        return False


FLASH_ATTENTION_AVAILABLE = check_flash_attention()


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
        print(f"测试下载速度时出现错误: {e}")
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


class LoadQwenOmniModel:
    def __init__(self):
        self.model_path = init_qwen_paths()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen2.5-Omni-7B",
                ], {"default": "Qwen2.5-Omni-7B"}),
                "quantization": ([
                    "👍 4-bit (VRAM-friendly)",
                    "⚖️ 8-bit (Balanced Precision)",
                    "🚫 None (Original Precision)"
                ], {"default": "👍 4-bit (VRAM-friendly)"})
            }
        }

    RETURN_TYPES = ("QWENOMNI", "OMNIPROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "🐼QwenOmni"

    def load_model(self, model_name, quantization):
        # 添加CUDA可用性检查
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for  {model_name} model")

        quant_config = None
        if quantization == "👍 4-bit (VRAM-friendly)":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "⚖️ 8-bit (Balanced Precision)":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # 自定义device_map，这里假设只有一个GPU，将模型尽可能放到GPU上
        device_map = {"": 0} if torch.cuda.device_count() > 0 else "auto"

        # 检查模型文件是否齐全
        if check_model_files_exist(self.model_path):
            print("模型文件已存在且齐全，无需下载。")
        else:
            # 测试下载速度
            huggingface_test_url = "https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main/model-00005-of-00005.safetensors"
            modelscope_test_url = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-Omni-7B/repo?Revision=master&FilePath=model-00005-of-00005.safetensors"
            huggingface_speed = test_download_speed(huggingface_test_url)
            modelscope_speed = test_download_speed(modelscope_test_url)

            if huggingface_speed >= modelscope_speed:
                download_sources = [
                    (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face"),
                    (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope")
                ]
            else:
                download_sources = [
                    (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope"),
                    (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face")
                ]

            max_retries = 3
            for download_func, repo_id, source in download_sources:
                for retry in range(max_retries):
                    print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）...")
                    try:
                        if download_func == snapshot_download:
                            download_func(
                                repo_id,
                                cache_dir=self.model_path,
                                ignore_patterns=["*.msgpack", "*.h5"]
                            )
                        else:
                            download_func(
                                repo_id,
                                cache_dir=self.model_path
                            )
                        print(f"成功从 {source} 下载模型。")
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试...")
                        else:
                            print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源...")
                else:
                    continue
                break
            else:
                raise RuntimeError("从所有源下载模型均失败。")

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            offload_state_dict=True,
            enable_audio_output=True,
        ).eval()

        # ✅ 编译优化（PyTorch 2.2+）
        if torch.__version__ >= "2.2":
            model = torch.compile(model, mode="reduce-overhead")

        # ✅ SDP优化（推荐）
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # 预加载模型到显存

        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        return model, processor


class QwenOmniParser:

    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": ("QWENOMNI",),
                "processor": ("OMNIPROCESSOR",),
                "prompt": ("STRING", {
                    "default": "Hi!😽",
                    "multiline": True
                }),
                "audio_output": ([
                    "🔇None (No Audio)",
                    "👱‍♀️Chelsie (Female)",
                    "👨‍🦰Ethan (Male)"
                ], {"default": "🔇None (No Audio)"}),
                "audio_source": ([
                    "🎧 Separate Audio Input",
                    "🎬 Video Built-in Audio"
                ],
                                 {
                                     "default": "🎧 Separate Audio Input",
                                     "display": "radio",
                                     "tooltip": "Select audio source: Use video's built-in audio track (priority) / Input a separate audio file (external audio)"
                                 }),
                "max_tokens": ("INT", {
                    "default": 128,
                    "min": 4,
                    "max": 2048,
                    "step": 16,
                    "display": "slider"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Higher values result in more random outputs. 0.1 - 0.3 is suitable for generating structured content."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                })
            },
            "optional": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "video_path": ("VIDEO_PATH",),
            }

        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "analyze_processor"
    CATEGORY = "🐼QwenOmni"

    def tensor_to_pil(self, image_tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @torch.no_grad()
    def analyze_processor(self, model, processor, prompt, max_tokens, temperature, audio_output, audio_source, top_p,
                          repetition_penalty, audio=None, video_path=None, image=None):
        pil_image = self.tensor_to_pil(image) if image is not None else None
        audio_path = None
        temp_audio_file = None

        if audio:
            try:
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
                audio_path = temp_audio_file.name
                waveform = audio["waveform"].squeeze(0).cpu().numpy()
                sample_rate = audio["sample_rate"]
                sf.write(audio_path, waveform.T, sample_rate)
            except Exception as e:
                print(f"Error saving audio to temporary file: {e}")
                audio_path = None

        SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": []}
        ]

        if pil_image is not None:
            conversation[-1]["content"].append({"type": "image", "image": pil_image})

        # 添加音频/视频输入（直接传递路径，由 qwen-omni-utils 处理）
        use_video_audio = audio_source == "🎬 Video Built-in Audio"
        if audio_path and not use_video_audio:
            conversation[-1]["content"].append({"type": "audio", "audio": audio_path})
        if video_path:
            conversation[-1]["content"].append({"type": "video", "video": video_path})  # 直接添加视频路径

        user_prompt = prompt if prompt.endswith(("?", ".", "！", "。", "？", "！")) else f"{prompt} "
        conversation[-1]["content"].append({"type": "text", "text": user_prompt})

        input_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        input_ids = processor.tokenizer(input_text, return_tensors="pt", padding=True)["input_ids"].to(model.device)
        input_length = input_ids.shape[1]

        processor_args = {
            "text": input_text,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": use_video_audio
        }

        # 直接调用 qwen-omni-utils 的多模态处理逻辑
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_video_audio)
        processor_args["audio"] = audios
        processor_args["images"] = images
        processor_args["videos"] = videos

        inputs = processor(**processor_args).to(model.device)

        generate_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "use_cache": True,
            "return_audio": audio_output != "🔇None (No Audio)",
            "use_audio_in_video": use_video_audio,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "pad_token_id": processor.tokenizer.pad_token_id
        }

        if generate_config["return_audio"]:
            generate_config["speaker"] = "Chelsie" if "Chelsie" in audio_output else "Ethan"

        outputs = model.generate(**inputs, **generate_config)

        # 统一批次维度，确保文本token是二维张量
        if generate_config["return_audio"]:
            text_tokens = outputs[0] if outputs[0].dim() == 2 else outputs[0].unsqueeze(0)
            audio_tensor = outputs[1]
        else:
            text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
            audio_tensor = torch.zeros(0, 0, device=model.device)

        # 严格截断输入提示，仅保留生成部分
        generated_ids = text_tokens[:, input_length:]

        text = processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # 去除生成提示符和用户prompt
        prefixes_to_remove = ["assistant", "ASSISTANT", "assistant:", "ASSISTANT:"]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip(" :\t\n")
                break
        user_prompt_clean = prompt.strip()
        if text.startswith(user_prompt_clean):
            text = text[len(user_prompt_clean):].strip()

        # 删除临时文件
        if temp_audio_file:
            try:
                os.remove(temp_audio_file.name)
            except Exception as e:
                print(f"Error deleting temporary audio file: {e}")
        if use_video_audio and video_audio_path:
            try:
                os.remove(video_audio_path)
            except Exception as e:
                print(f"Error deleting video audio temp file: {e}")

        # 处理音频部分（不变）
        if generate_config["return_audio"]:
            audio = audio_tensor
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).to(model.device)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        else:
            audio = torch.zeros(0, 0, device=model.device)

        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        assert audio.dim() == 2, f"Audio waveform must be 2D, got {audio.dim()}D"

        audio_output_data = {
            "waveform": audio,
            "sample_rate": 24000
        }

        if generate_config["return_audio"]:
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_output_data["waveform"].cpu(), 24000, format="wav")
            buffer.seek(0)
            waveform, sample_rate = torchaudio.load(buffer)
            audio_output_data = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }

        del outputs
        torch.cuda.empty_cache()

        return (text, audio_output_data)


NODE_CLASS_MAPPINGS = {
    "VideoUploader": VideoUploader,
    "LoadQwenOmniModel": LoadQwenOmniModel,
    "QwenOmniParser": QwenOmniParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUploader": "Video Uploader🐼",
    "LoadQwenOmniModel": "Load Qwen Omni Model🐼",
    "QwenOmniParser": "Qwen Omni Parser🐼",
}