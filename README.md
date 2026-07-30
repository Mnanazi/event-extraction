# event-extraction

基于 RoBERTa + CRF 的中文句子级事件抽取模型。采用联合标注体系，在 DuEE 1.0 数据集 Top 10 高频事件上训练，实现触发词识别与论元抽取。
利用AI编程工具进行辅助，实际使用中发现网页端千问的`Qwen3.7-Max`适合整体项目规划、针对单个python文件的优化、big修复，Github Copilot 适合项目bug修复、跨文件的局部代码优化（尤其是在bug修复时） 。

## 快速开始

### 环境要求

- Python 3.12+
- CUDA 12.6
- 操作系统：Ubuntu 24.04.2 LTS（WSL2）
- 显存：4GB（GTX 1650 已验证）

### 安装

```bash
git clone <你的仓库地址>
cd event-extraction
uv sync
-Max