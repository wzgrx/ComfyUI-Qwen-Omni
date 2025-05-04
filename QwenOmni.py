from __future__ import annotations
import torch
import os
import tempfile
import io
import torchaudio
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
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
    """初始化模型路径，确保使用绝对路径"""
    base_dir = Path(folder_paths.models_dir).resolve()
    qwen_dir = base_dir / "Qwen" # 添加VLM子目录如 / "Qwen" / "VLM"
    model_dir = qwen_dir / "Qwen2.5-Omni-7B"    

    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册到ComfyUI
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(model_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(model_dir)], {'.safetensors', '.bin'})
    
    print(f"模型路径已初始化: {model_dir}")
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


def validate_model_path(model_path):
    """验证模型路径的有效性和模型文件是否齐全"""
    path_obj = Path(model_path)
    
    # 基本路径检查
    if not path_obj.is_absolute():
        print(f"错误: {model_path} 不是绝对路径")
        return False
    
    if not path_obj.exists():
        print(f"模型目录不存在: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"错误: {model_path} 不是目录")
        return False
    
    # 检查模型文件是否齐全
    if not check_model_files_exist(model_path):
        print(f"模型文件不完整: {model_path}")
        return False
    
    return True




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
        # 重置环境变量，避免干扰
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)     

        self.model_path = init_qwen_paths()
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"模型路径: {self.model_path}")
        print(f"缓存路径: {self.cache_dir}")
        
        # 验证并创建缓存目录
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model = None
        self.processor = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["Qwen2.5-Omni-7B"],
                    {
                        "default": "Qwen2.5-Omni-7B",
                        "tooltip": "Select the available model version. Currently, only the Qwen2.5-Omni-7B multimodal large model is supported."
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage, suitable for resource-constrained environments.\n⚖️ 8-bit: Strikes a balance between precision and performance.\n🚫 None: Uses the original floating-point precision (requires a high-end GPU)."
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Hi!😽",
                        "multiline": True,
                        "tooltip": "Enter a text prompt, supporting Chinese and emojis. Example: 'Describe a cat in a painter's style.'"
                    }
                ),
                "audio_output": (
                    [
                        "🔇None (No Audio)",
                        "👱‍♀️Chelsie (Female)",
                        "👨‍🦰Ethan (Male)"
                    ],
                    {
                        "default": "🔇None (No Audio)",
                        "tooltip": "Audio output options:\n🔇 Do not generate audio.\n👱‍♀️ Use the female voice Chelsie (warm tone).\n👨‍🦰 Use the male voice Ethan (calm tone)."
                    }
                ),
                "audio_source": (
                    [
                        "🎧 Separate Audio Input",
                        "🎬 Video Built-in Audio"
                    ],
                    {
                        "default": "🎧 Separate Audio Input",
                        "display": "radio",
                        "tooltip": "Select audio source: Use video's built-in audio track (priority) / Input a separate audio file (external audio)"
                    }
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 132,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "display": "slider",
                        "tooltip": "Control the maximum length of the generated text (in tokens). \nGenerally, 100 tokens correspond to approximately 50 - 100 Chinese characters or 67 - 100 English words, but the actual number may vary depending on the text content and the model's tokenization strategy. \nRecommended range: 64 - 512."
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Control the generation diversity:\n▫️ 0.1 - 0.3: Generate structured/technical content.\n▫️ 0.5 - 0.7: Balance creativity and logic.\n▫️ 0.8 - 1.0: High degree of freedom (may produce incoherent content)."
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Nucleus sampling threshold:\n▪️ Close to 1.0: Retain more candidate words (more random).\n▪️ 0.5 - 0.8: Balance quality and diversity.\n▪️ Below 0.3: Generate more conservative content."
                    }
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Control of repeated content:\n⚠️ 1.0: Default behavior.\n⚠️ >1.0 (Recommended 1.2): Suppress repeated phrases.\n⚠️ <1.0 (Recommended 0.8): Encourage repeated emphasis."
                    }
                )
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Upload a reference image (supports PNG/JPG), and the model will adjust the generation result based on the image content."
                    }
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Upload an audio file (supports MP3/WAV), and the model will analyze the audio content and generate relevant responses."
                    }
                ),
                "video_path": (
                    "VIDEO_PATH",
                    {
                        "tooltip": "Enter the video file  (supports MP4/WEBM), and the model will extract visual features to assist in generation."
                    }
                )
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "process"
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



        # 检查模型文件是否存在且完整
        if not validate_model_path(self.model_path):
            print(f"检测到模型文件缺失，正在为你下载 {model_name} 模型，请稍候...")
            print(f"下载将保存在: {self.model_path}")
            
            # 开始下载逻辑
            try:
                # 测试下载速度
                huggingface_test_url = "https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main/model-00005-of-00005.safetensors"
                modelscope_test_url = "https://modelscope.cn/api/v1/models/qwen/Qwen2.5-Omni-7B/repo?Revision=master&FilePath=model-00005-of-00005.safetensors"
                huggingface_speed = test_download_speed(huggingface_test_url)
                modelscope_speed = test_download_speed(modelscope_test_url)


                print(f"Hugging Face下载速度: {huggingface_speed:.2f} KB/s")
                print(f"ModelScope下载速度: {modelscope_speed:.2f} KB/s")

                # 优化判断条件：只有当Hugging Face速度超过ModelScope 50%时才优先选择

                if huggingface_speed > modelscope_speed * 1.5:
                    download_sources = [
                        (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face"),
                        (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope")
                    ]
                    print("基于下载速度分析，优先尝试从Hugging Face下载")
                else:
                    download_sources = [
                        (modelscope_snapshot_download, "qwen/Qwen2.5-Omni-7B", "ModelScope"),
                        (snapshot_download, "Qwen/Qwen2.5-Omni-7B", "Hugging Face")
                    ]
                    print("基于下载速度分析，优先尝试从ModelScope下载")

                max_retries = 3
                success = False
                final_error = None
                used_cache_path = None

                for download_func, repo_id, source in download_sources:
                    for retry in range(max_retries):
                        try:
                            print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）...")
                            if download_func == snapshot_download:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    ignore_patterns=["*.msgpack", "*.h5"]
                                )
                            else:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir
                                )

                            used_cache_path = cached_path  # 记录使用的缓存路径
                            
                            # 将下载的模型复制到模型目录
                            self.copy_cached_model_to_local(cached_path, self.model_path)
                            
                            print(f"成功从 {source} 下载模型到 {self.model_path}")

                            # 下载成功提示
                            print("\n⚠️ 注意：模型下载过程中使用了缓存文件")
                            print(f"缓存路径: {cached_path}")
                            print("为避免占用额外硬盘空间，你可以在确认模型正常工作后删除此缓存目录")
                            
                            success = True
                            break

                        except Exception as e:
                            final_error = e  # 保存最后一个错误
                            if retry < max_retries - 1:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试...")
                            else:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源...")
                    if success:
                        break
                else:
                    raise RuntimeError("从所有源下载模型均失败。")
                
                # 下载完成后再次验证
                if not validate_model_path(self.model_path):
                    raise RuntimeError(f"下载后模型文件仍不完整: {self.model_path}")
                
                print(f"模型 {model_name} 已准备就绪")
                
            except Exception as e:
                print(f"下载模型时发生错误: {e}")
                
                # 下载失败提示
                if used_cache_path:
                    print("\n⚠️ 注意：下载过程中创建了缓存文件")
                    print(f"缓存路径: {used_cache_path}")
                    print("你可以前往此路径删除缓存文件以释放硬盘空间")
                
                raise RuntimeError(f"无法下载模型 {model_name}，请手动下载并放置到 {self.model_path}")

        # 模型文件完整，正常加载
        print(f"加载模型: {self.model_path}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
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
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # ✅ SDP优化（推荐）
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def tensor_to_pil(self, image_tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @torch.no_grad()
    def process(self, model_name, quantization, prompt, audio_output, audio_source, max_tokens, temperature, top_p,
                repetition_penalty, audio=None, image=None, video_path=None):
        if self.model is None or self.processor is None:
            self.load_model(model_name, quantization)

        pil_image = None
        if image is not None:
            pil_image = self.tensor_to_pil(image)
            max_res = 1024
            if max(pil_image.size) > max_res:
                pil_image.thumbnail((max_res, max_res))
                pil_image = np.array(pil_image)
                pil_image = torch.from_numpy(pil_image).permute(2, 0, 1).unsqueeze(0) / 255.0
                pil_image = Image.fromarray((pil_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

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

        input_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

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

        inputs = self.processor(**processor_args).to(self.model.device)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if v is not None
        }

        generate_config = {
            "max_new_tokens": max(max_tokens, 10),
            "temperature": temperature,
            "do_sample": True,
            "use_cache": True,
            "return_audio": audio_output != "🔇None (No Audio)",
            "use_audio_in_video": use_video_audio,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if generate_config["return_audio"]:
            generate_config["speaker"] = "Chelsie" if "Chelsie" in audio_output else "Ethan"

        outputs = self.model.generate(**model_inputs, **generate_config)

        # 统一批次维度，确保文本token是二维张量
        if generate_config["return_audio"]:
            text_tokens = outputs[0] if outputs[0].dim() == 2 else outputs[0].unsqueeze(0)
            audio_tensor = outputs[1]
        else:
            text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
            audio_tensor = torch.zeros(0, 0, device=self.model.device)

        # 关键修正：对 text_tokens 进行 token 切片处理
        input_length = model_inputs["input_ids"].shape[1]
        text_tokens = text_tokens[:, input_length:]  # 截取新生成的token

        # 直接获取完整的生成文本
        text = self.tokenizer.decode(
            text_tokens[0],  # 使用正确的变量
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 删除临时文件
        if temp_audio_file:
            try:
                os.remove(temp_audio_file.name)
            except Exception as e:
                print(f"Error deleting temporary audio file: {e}")
        if use_video_audio and 'video_audio_path' in locals():
            try:
                os.remove(video_audio_path)
            except Exception as e:
                print(f"Error deleting video audio temp file: {e}")

        # 处理音频部分（不变）
        if generate_config["return_audio"]:
            audio = audio_tensor
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).to(self.model.device)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        else:
            audio = torch.zeros(0, 0, device=self.model.device)

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

        return (text.strip(), audio_output_data)





NODE_CLASS_MAPPINGS = {
    "VideoUploader": VideoUploader,
    "QwenOmniCombined": QwenOmniCombined
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUploader": "Video Uploader🐼",
    "QwenOmniCombined": "Qwen Omni Combined🐼"
}
    