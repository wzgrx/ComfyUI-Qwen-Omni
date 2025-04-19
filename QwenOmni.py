import torch
import os
import io
from io import BytesIO
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_omni_utils import process_mm_info
import numpy as np
import soundfile as sf
import re
import datetime
import torchaudio
device = "cuda"

def check_flash_attention():
    """Check if flash attention 2 is available"""
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False

FLASH_ATTENTION_AVAILABLE = check_flash_attention()

def init_qwen_paths():
    """动态初始化模型路径"""
    qwen_base = Path(folder_paths.models_dir) / "Qwen"
    qwen_model_dir = qwen_base / "Qwen2.5-Omni-7B"
    
    # 创建必要目录
    qwen_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册路径到系统
    if not hasattr(folder_paths, "add_model_folder_path"):
        # 兼容旧版本手动注册
        if "Qwen" not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["Qwen"] = ([str(qwen_model_dir)], {'.safetensors', '.bin'})
    else:
        folder_paths.add_model_folder_path("Qwen", str(qwen_model_dir))
    
    return str(qwen_model_dir)


class LoadQwenOmniModel:
    def __init__(self):
        self.model_path = init_qwen_paths()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
            }
        }

    RETURN_TYPES = ("QWENOMNI", "OMNIPROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "🐼QwenOmni"
  
    def load_model(self):
        # 添加Flash Attention支持判断
        attn_implementation = "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else None
        # 添加量化配置
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 改用FP16加速计算
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,        # 打开or关闭双重量化
            llm_int8_threshold=6.0,
        )

        


        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 使用更高效的BF16格式
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # 启用更快的safetensors格式
            offload_state_dict=True,  # 优化显存分配
            enable_audio_output=True
        ).eval()

        
        # 预加载模型到显存
        
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        return model, processor

class QwenOmniParser:

    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWENOMNI",),
                "processor": ("OMNIPROCESSOR",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail", 
                    "multiline": True
                }),
                "max_tokens": ("INT", {
                    "default": 128, 
                    "min": 32, 
                    "max": 512
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "audio_mode": ([
                    "🔇None (No Audio)", 
                    "👱‍♀️Chelsie (Female)", 
                    "👨🏻Ethan (Male)"
                ], {"default": "🔇None (No Audio)"}),
            }
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text", "audio")
    FUNCTION = "analyze_image"
    CATEGORY = "🐼QwenOmni"

    def tensor_to_pil(self, image_tensor):
        """优化图像张量转换"""
        # 处理批次维度 [B x H x W x C]
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
            
        # 数值范围转换 [0-1] => [0-255]
        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        return Image.fromarray(image_np)

    @torch.no_grad()
    def analyze_image(self, model, processor, image, prompt, max_tokens, temperature, audio_mode):
        # 转换输入格式
        pil_image = self.tensor_to_pil(image)
        
        # 定义系统提示常量
        DEFAULT_SYSTEM_PROMPT = "AI Assistant"
        OFFICIAL_AUDIO_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

        # 解析音频参数
        enable_audio = audio_mode != "🔇None (No Audio)"
        voice_type = "Chelsie" if "Chelsie" in audio_mode else "Ethan" if "Ethan" in audio_mode else None

        # 阶段一：生成核心文本
        def generate_core_text():
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            inputs = processor(
                text=processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False), # 关闭自动添加角色提示
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            generate_config = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "temperature": temperature,
                "use_cache": True,
                "return_audio": False
            }
            text_ids = model.generate(**inputs, **generate_config)
            return processor.batch_decode(text_ids, skip_special_tokens=True)[0]

        # 阶段二：基于文本生成语音
        def generate_speech(text):
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": OFFICIAL_AUDIO_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"<|im_start|>user\n{text}<|im_end|>"}  # ✅ 使用原始文本标记
                ]}
            ]
            inputs = processor(
                text=processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False), # 关闭自动添加角色提示
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            # 智能配置生成参数 ▼▼▼
            generate_config = {
                "max_new_tokens": len(text.split()) * 3,
                "do_sample": False,
                "use_cache": True,
                "return_audio": True
            }
            
            # 有效性验证后添加发音人参数 ▼▼▼
            if voice_type in {"Chelsie", "Ethan"}:  # 使用集合加速判断
                generate_config["speaker"] = voice_type
            else:
                print(f"[WARN] 使用模型默认发音人，当前选择: {audio_mode}")
            
            _, audio = model.generate(**inputs, **generate_config)
            return audio

        # 主流程
        text = generate_core_text()
        audio = torch.zeros(0)
        
        if enable_audio:
            # 二次验证发音人有效性 ▼▼▼
            if voice_type is None:
                raise ValueError(f"无效的发音人配置，audio_mode: {audio_mode}")
            audio = generate_speech(text)

        return (text, audio)
    


class SaveQwenOmniAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "output.wav"}),
                "samplerate": ("INT", {"default": 23000}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "🐼QwenOmni"    
    def save_audio(self, audio, filename, samplerate):
        # 获取ComfyUI的输出目录
        output_dir = folder_paths.get_output_directory()
        # 生成日期部分（yyyyMMdd）
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        # 查找当天最新序号
        existing_files = os.listdir(output_dir)
        pattern = re.compile(rf"^{date_str}_(\d{{4}})\.wav$")
        # 提取已有序号并找到最大值
        max_sequence = 0
        for filename in existing_files:
            match = pattern.match(filename)
            if match:
                current_seq = int(match.group(1))
                max_sequence = max(max_sequence, current_seq)
        
        # 生成新序号（自动递增）
        new_sequence = max_sequence + 1
        
        # 确保文件名不包含路径（防止目录注入）
        filename = os.path.basename(filename)
        # 构建完整文件名
        new_filename = f"{date_str}_{new_sequence:04d}.wav"
        full_path = os.path.join(output_dir, new_filename)
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # # 构建完整保存路径
        # full_path = os.path.join(output_dir, filename)
        
        # # 创建目录（如果不存在）
        # os.makedirs(output_dir, exist_ok=True)
        
        # 保存音频文件
        sf.write(
            full_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=samplerate,
        )
        # return ()
        return {"ui": {"audio": [full_path]}}



NODE_CLASS_MAPPINGS = {
    "LoadQwenOmniModel": LoadQwenOmniModel,
    "QwenOmniParser": QwenOmniParser,
    "SaveQwenOmniAudio": SaveQwenOmniAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenOmniModel": "Load QwenOmni Model🐼",
    "QwenOmniParser": "QwenOmni Parser🐼",
    "SaveQwenOmniAudio": "Save QwenOmni Audio🐼",
}