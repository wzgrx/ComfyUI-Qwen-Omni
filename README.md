
# ComfyUI-Qwen-Omni 🐼

**基于 Qwen2.5-Omni-7B 的多模态大型语言模型 ComfyUI 插件**



🔄 端到端多模态交互
ComfyUI-Qwen-Omni 是首个支持端到端多模态交互的 ComfyUI 插件，实现了文本、图像、音频的无缝联合生成与编辑。无需中间步骤，只需一次操作，即可让模型同时理解并处理多种输入模态，生成连贯的文本描述和语音输出，为 AI 创作提供前所未有的流畅体验。



这个插件将 Qwen2.5-Omni-7B 多模态大模型集成到 ComfyUI 中，支持文本、图像、音频和视频输入，并能生成文本和语音输出，为您的 AI 创作提供更丰富的交互体验。


## 🌟 特性亮点

- **多模态输入**：支持文本、图像、音频和视频作为输入
- **文本生成**：基于多模态输入生成连贯的文本描述
- **语音合成**：支持生成自然流畅的语音输出（男声/女声可选）
- **参数化控制**：可调整温度、最大 tokens、采样策略等生成参数
- **GPU 优化**：支持 4-bit/8-bit 量化，降低显存需求


## 🚀 安装方法

1. **克隆仓库到 ComfyUI 扩展目录**：

   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-Qwen-Omni.git
   cd ComfyUI-Qwen-Omni
   pip install -r requirements.txt
   ```

2. **下载模型文件**：

   插件会在首次运行时自动下载 Qwen2.5-Omni-7B 模型，或您也可以提前手动下载并放置到 `ComfyUI/models/Qwen/Qwen2.5-Omni-7B/` 目录下。


## 📖 使用指南

1. 在 ComfyUI 中添加 "Qwen Omni Combined" 节点
2. 配置参数：
   - 选择量化方式（4-bit/8-bit/不量化）
   - 输入文本提示
   - 选择是否生成语音及语音类型
   - 调整生成参数（温度、最大 tokens 等）
3. 可选：连接图像、音频或视频输入
4. 执行节点生成结果


## 🎛️ 参数说明

| 参数               | 描述                                                                 |
|--------------------|----------------------------------------------------------------------|
| `max_tokens`       | 控制生成文本的最大长度（以 token 为单位）。通常，100 个 token 大约对应 50 - 100 个汉字或 67 - 100 个英文单词。 |
| `temperature`      | 控制生成多样性：较低值生成更结构化的内容，较高值生成更随机的内容。           |
| `top_p`            | 核采样阈值，控制词汇选择范围：接近 1 保留更多候选词，较小值生成更保守内容。 |
| `repetition_penalty` | 控制重复内容：>1 抑制重复，<1 鼓励重复强调。                              |
| `quantization`     | 模型量化选项：4-bit（显存友好）、8-bit（平衡精度）或不量化（高精度）。    |
| `audio_output`     | 语音输出选项：不生成语音、女声（Chelsie）或男声（Ethan）。               |


## 💡 使用示例

### 图像描述生成
1. 输入图像
2. 文本提示："Describe this image in detail"
3. 输出详细的图像描述文本

### 视频内容分析
1. 输入视频
2. 文本提示："What's happening in this video?"
3. 输出视频内容的文字描述和分析


## ⚙️ 技术细节

- **模型**：Qwen2.5-Omni-7B
- **框架**：基于 Transformers 库
- **优化**：支持 Flash Attention 2 加速
- **显存要求**：4-bit 量化下约需 10GB GPU 显存


## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！请提交 Pull Request 或创建 Issue。


## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。


## 👀 预览图

![Qwen Omni in ComfyUI](https://picsum.photos/800/450)

*插件在 ComfyUI 中的使用界面示例*


## 📧 联系我

如有问题或建议，请创建 GitHub Issue 或联系我：[503887319@qq.com](mailto:503887319@qq.com)


贡献指南 🤝
我们欢迎以下形式的贡献：

✨ 新功能提案
🐛 问题报告（请包含复现步骤和日志）
📝 功能改进
🖼️ 示例工作流

---

以下是为您的项目 README 添加的 **致谢部分**，您可以将其放在合适的位置（建议在“特性亮点”或“技术细节”之后）：


## 🙏 致谢

我们衷心感谢以下团队和项目对 ComfyUI-Qwen-Omni 开发的支持与贡献：

### 🌟 核心技术支持
- **Qwen 团队（阿里巴巴集团）**  
  感谢 Qwen-Omni 系列模型的开发者，特别是 **Qwen2.5-Omni-7B** 模型的开源贡献。  
  他们在多模态大模型领域的突破性工作，为插件提供了强大的底层能力支持。  
  [Qwen 官方项目](https://github.com/QwenLM)

- **豆包团队（字节跳动）**  
  在插件开发过程中，豆包 AI 在代码调试、文档生成和问题排查中提供了重要帮助，极大提升了开发效率。  
  [豆包官网](https://doubao.com)

### 🛠️ 工具与框架
- **Hugging Face Transformers**  
  感谢其提供的高效模型加载与处理工具，简化了多模态集成流程。  
  [Hugging Face Transformers](https://github.com/huggingface/transformers)

- **ComfyUI 社区**  
  ComfyUI 灵活的节点化架构为插件开发提供了理想的生态环境。  
  [ComfyUI 项目](https://github.com/comfyanonymous/ComfyUI)

### 👥 开源精神
感谢所有开源社区的贡献者，正是你们的分享让技术进步成为可能。  
如果您在使用中发现问题或有改进建议，欢迎通过 GitHub Issue 与我们交流！


### 如何引用我们
如果您在研究或项目中使用了 ComfyUI-Qwen-Omni，欢迎在 README 或论文中引用我们：


@software{ComfyUI-Qwen-Omni,
  author = {Xiong Song},
  title = {ComfyUI-Qwen-Omni: End-to-End Multimodal Plugin for ComfyUI},
  url = {https://github.com/SXQBW/ComfyUI-Qwen-Omni},
  version = {0.1.0},
  year = {2025}
}

这样的致谢既表达了对核心团队的感谢，也突出了项目的技术依赖和开源生态，同时引导用户正确引用项目。记得将 `yourusername` 替换为您的 GitHub 用户名哦！