# 给 AI漫画编剧大师 做贡献

仓库目录名是 `yizi`。产品名是 **AI漫画编剧大师**。

不生新页，不生图。图不上网。不要把本机 `user_data/` 提交上来。

## 本地跑

Windows 10/11，Python 3.11+。

```bat
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\python.exe -m yizi
```

浏览器打开 http://127.0.0.1:8766

改 Python 后要重启进程。改 `frontend/` 后硬刷新（地址栏带新的 `?v=`）。

## 测试

```bat
set YIZI_VLM=0
set YIZI_DETECT=0
.venv\Scripts\python.exe -m pytest -q
```

不要在测试里连真实视觉模型，也不要读取你自己的工程稿。

## 改什么时注意

- 主路径是：导入成页 → 检出框 → 看图出场景卡 → 编剧写词 → 嵌进原框
- 编剧接口只收场景卡、大纲和框的版式，不收原图
- 中文停在独立文字层，不要烤进 `source.png`
- 健康检查 JSON 保持 `{"ok":true,"name":"yizi"}`，不要往里面塞字段，除非同步改测试

## 提交

PR 说清楚为什么。不要带 Key、成页、`.gguf`、`.onnx`、zip 成品。Issue 和 PR 用中文即可。
