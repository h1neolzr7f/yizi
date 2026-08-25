# 译字

成页已经画好了。还差能读的中文。

**AI漫画编剧大师**是给已经出好页的人用的本机工具。把漫画拖进来，写几句你想讲的：它先在这台电脑上看图，再按页上**已有的框**写出对白、旁白和余韵。

不另生新页。不另生图。图不上网。

[下载 1.0 源码包](https://github.com/h1neolzr7f/yizi/releases/tag/v1.0.0) · MIT

## 它做什么

你手里已经有成页——自己画的、模型出的、扫的都成。缺的是顺着画面读下去的中文。

1. 导入成页或整个文件夹，排出阅读序
2. 写几句话，按 Pixiv 短篇口吻扩写定调
3. 本机检出文字框，看图只填场景卡
4. 按原框回填台词，改完再导出

主路径：**导入成页 → 检出框 → 看图出场景卡 → 编剧写词 → 嵌进原框**

## 它不做什么

- 不替你生成新页、新图、新分镜
- 不把原图发给编剧接口；接口只收场景卡、大纲和框的版式
- 不是网页产品。打开的是本机 `http://127.0.0.1:8766`

没填 Key 也能走完，只用本机草稿，界面会写明。中文停在独立文字层，不会烤进 `source.png`。

## 安装

Windows 10/11，Python 3.11+。

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

可选：NVIDIA 显卡做本机看图；OpenAI 兼容 Key 做编剧（官方 DeepSeek 等）。两者都不是开工条件。

## 本机视觉（可选）

图仍然只在本机推理。

```bat
.venv\Scripts\python.exe scripts\setup_vision.py
```

模型没就绪时，自动退回几何反推，整条链路还能走完。

## 使用

1. 把成页或文件夹拖进去
2. 右边写几句你想讲的，不必完整
3. 点「用这几句话回填全部」：按阅读序看图、写词、铺层
4. 双击框改词，拖角改大小，满意后导出

设置里的 Key、工程、模型都在本地 `user_data/`。这个目录**不会**进 Git，也请不要提交上去。

## 开源

- 许可证：MIT，见 `LICENSE`
- 不要把 `user_data/`、API Key、成页原图提交到仓库
- 检出模型 Manga-Bubble-YOLO（Apache-2.0）；视觉模型请遵守各自许可证

## 测试

```bat
set YIZI_VLM=0
set YIZI_DETECT=0
.venv\Scripts\python.exe -m pytest -q
```
