# AI漫画编剧大师（yizi）1.0

成页漫画进来，按页上**已有的框**设计情节、写出能读的中文。不生新页，不生图。

看图只在本机做，图不上网。编剧接口只收场景卡、大纲和框的版式，不收原图。

## 能做什么

1. 导入成页或整个文件夹，排出阅读序  
2. 写几句话，按 Pixiv 短篇口吻扩写定调  
3. 检出文字框，本机看图只填场景卡  
4. 按已有框逐页回填台词，改完再导出  

主路径：**导入成页 → 检出框 → 看图出场景卡 → 编剧写词 → 嵌进原框**

## 要求

- Windows 10/11，Python 3.11+  
- 浏览器打开本机服务（默认 `http://127.0.0.1:8766`）  
- 可选：NVIDIA 显卡 + 本机视觉模型（Qwen3-VL）  
- 可选：OpenAI 兼容编剧 Key（官方 DeepSeek 等）  

## 安装

```bat
cd /d 你的目录\yizi
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

双击 `start.bat`，或：

```bat
.venv\Scripts\python.exe -m yizi
```

浏览器打开 http://127.0.0.1:8766

本机视觉（可选，图仍不上网）：

```bat
.venv\Scripts\python.exe scripts\setup_vision.py
```

模型未就绪时自动退回几何反推，整条链路仍能走完。

## 使用

1. 把成页或文件夹拖进去  
2. 右边写几句你想讲的  
3. 点「用这几句话回填全部」：按阅读序检出、看图、写词、铺层  
4. 双击框改词，拖角改大小，满意后导出  

不填 Key 也能走完，只用本机草稿，界面会标明。中文停在独立文字层，不会烤进 `source.png`。

设置里的 Key、工程、模型都在本地 `user_data/`，这个目录**不会**进 Git。

## 开源说明

- 许可证：MIT，见 `LICENSE`  
- 不要把 `user_data/`、API Key、成页原图提交到仓库  
- 检出模型 Manga-Bubble-YOLO（Apache-2.0）；视觉模型请遵守各自许可证  

## 测试

```bat
set YIZI_VLM=0
set YIZI_DETECT=0
.venv\Scripts\python.exe -m pytest -q
```
