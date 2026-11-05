# 双声道：VAD → ASR → 拼音 / Qwen 字级对齐 → Praat

更新日期：2026-10-30。主程序：`vad_asr_align_praat.py`。

## 1. 数据关系
两个声道独立处理、统一时间轴导出。

原始文件是 48 kHz、Opus 编码、双声道。容器显示约 39.271 秒；实际解码并重采样后的音频为 627840 个采样点、16 kHz，即 **39.24 秒**。所有输出使用实际解码采样点作为时间基准。两路不是重复信号；分别标为 `ch1`、`ch2`。

```text
双声道 WebM → FFmpeg 一次解码为 16 kHz 双声道 WAV
                    ├─ ch1 全长 WAV → VAD 切片 → ASR 汉字 → 整句拼音
                    │                            └─ 音频 + 汉字 → Qwen 对齐
                    └─ ch2 全长 WAV → 相同处理                    ↓
                                                  拼音使用对应汉字的边界
                                                              ↓
                             两路各自 TextGrid + 合并的 8 层 TextGrid
```

说明：

- WebM 先由 FFmpeg 解码。
- 不下混声道，不剪掉全长录音中的静音，不拼接切片；两个声道共同从 0 开始。
- **Qwen 输入是录音及汉字文本，不是拼音。** 拼音在整句语境下生成，然后映射到对齐单元。

## 2. 环境与模型

首次安装示例：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements_audio.txt
.venv/bin/python -m pip check
```

`imageio-ffmpeg` 提供本地 FFmpeg。系统 PATH 中缺少 FFmpeg 时，FunASR 可能给出提示；本程序仍会先通过该独立二进制正确解码 WebM。

主程序使用 CUDA（可用时）或 CPU。GPU 环境可按对应平台重新安装 torch / torchaudio；依赖文件记录的是本次运行环境的直接依赖版本，不是所有平台唯一的安装组合。

FunASR 首次运行自动下载 `fsmn-vad`、`paraformer-zh`、`ct-punc`，默认缓存放在本项目 `models/funasr`。可事先设置 `MODELSCOPE_CACHE` 指向已有缓存。

Qwen 模型使用本地目录，首次下载一次：

```bash
.venv/bin/python -c 'from modelscope import snapshot_download; snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B", local_dir="models/Qwen3-ForcedAligner-0.6B")'
```

也可以将脚本中的 `ALIGNER_PATH` 改为其他已有模型目录，或者 `"Qwen/Qwen3-ForcedAligner-0.6B"` 让 Hugging Face 加载。

## 3. 执行步骤

脚本顶部已经填写本次文件路径和输出目录：

```python
INPUT_AUDIO = BASE_DIR / "【人AI语音交互_pilot_V2_打乱_AI BC人】CY-1029 1759-第9题-题号63-正式-a1-实际.webm"
OUTPUT_DIR = BASE_DIR / "outputs" / "CY-1029_Q63"
CHANNELS = (0, 1)
STAGE = "all"
```

执行：

```bash
# 已经对齐的录音：仅更新空白标签和 CSV，省去模型推理
.venv/bin/python vad_asr_align_praat.py "outputs/CY-1029_Q63" --stage export

# 新的双声道录音：指定空输出目录，运行完整流程
.venv/bin/python vad_asr_align_praat.py "outputs/新录音" --audio "/实际路径/新录音.webm"
```

| STAGE | 处理 | 主要结果 |
| --- | --- | --- |
| `prepare` | 一次解码并重采样，保留双声道 | 根目录 `recording.wav` 和 `recording.json` |
| `vad` | 准备音频，对两路分别 VAD、保存切片 | `ch1` / `ch2` 下的全长 WAV、切片、`segments.json` |
| `asr` | 识别已有切片、整句转拼音 | 更新两个声道的 `segments.json` |
| `align` | Qwen 对齐当前汉字文本、映射拼音、导出 | 每段 / 每声道 / 合并 TextGrid 和 CSV |
| `all` | 顺序执行准备、VAD、ASR、对齐 | 全部结果 |

如单独执行一个阶段：

```bash
.venv/bin/python -c 'import vad_asr_align_praat as p; p.STAGE="align"; p.main()'
```

**推荐人工校对顺序：**

1. 执行 `vad`，再执行 `asr`。
2. 打开 `ch1/segments.json` 和 `ch2/segments.json`，对照切片音频修改各段的 `text`。保留真实重复、语气词和口头语；将数字按实际读音写成汉字。`asr_text` 保留原始机器识别结果。
3. 执行 `align`。它会按最新 `text` 重算拼音并重新生成 TextGrid。
4. 在 Praat 检查文字与字边界，优先处理 `review` 点层。

重跑 `all` / `vad` 会复用已有 VAD 清单；重跑 `asr` 会跳过 `asr_done=true` 或已有文本的片段，保留人工修改。空识别也标为完成，避免反复识别同一空结果。要重新识别某段，应清空该段 `asr_text` 和 `text`，并设 `asr_done=false`。

**更换录音、覆盖同名输入文件或修改 VAD 参数时，使用新的输出目录。** 程序不计算文件内容指纹，也不自动判断同一路径下的音频内容是否已改变。重新 `align` 会覆盖机器生成的 TextGrid；已在 Praat 手动改过的版本请另存。

## 4. 输出

```text
outputs/CY-1029_Q63/
    recording.wav               # 双声道、16 kHz、39.24 秒
    recording.json              # 输入来源、声道数、实际采样点数
    recording.TextGrid          # 两路合并、8 层、共同全局时间轴
    units.csv                   # 两路全部单元，channel 列为 1 / 2
    ch1/
        recording.wav           # 第一声道全长音频
        recording.TextGrid      # 第一声道的 4 层标注
        segments.json           # VAD 起止采样点、ASR、可校对文本、拼音
        units.csv
        clips/
            0001.wav
            0001.TextGrid       # 本切片内时间，从 0 开始
    ch2/
        ...                     # 第二声道的相同输出结构
```

合并 TextGrid 的层名：`ch1_segments`、`ch1_hanzi`、`ch1_pinyin`、`ch2_segments`、`ch2_hanzi`、`ch2_pinyin`、`ch1_review`、`ch2_review`。

| 层 | 类型 | 内容 |
| --- | --- | --- |
| `segments` | IntervalTier | VAD 片段范围和文本，包括句内标点 |
| `hanzi` | IntervalTier | 有效汉字边界；混合英文通常以词为单位 |
| `pinyin` | IntervalTier | 与汉字相同的边界，整句生成的拼音 |
| `review` | TextTier | 空转写、零时长、逆序、重叠、越界等待复核项目 |

## 5. 在 Praat 中打开文件
在 Praat 中 **Open → Read from file...** 读取根目录的 `recording.wav` 和 `recording.TextGrid`，同时选中 Sound 和 TextGrid，点击 **View & Edit**。这样可以同时看两路波形及各自标注。只检查一路时，打开相应 `ch1` / `ch2` 中的同名文件。

`units.csv` 包含 `channel`、`segment_id`、`unit_index`、`text`、`pinyin`、切片内起止时间、全局起止时间和 `status`，时间单位均为秒：

```text
全局时间 = 片段 start_sample / sample_rate + 切片内预测时间
```

`status=ok` 只表示区间结构合法，不代表边界准确率已经人工验证。其他值为 `zero_duration`、`reversed`、`outside_clip`、`overlap`、`nonfinite`，可能组合出现。异常单元不进入正常汉字/拼音区间层，但仍完整保留在 CSV 与复核点中。空白区间不统一标为静音，因为也可能是遗漏或边界不明的语音。

## 6. 方法边界

- **文本错误会传递给对齐。** 强制对齐不会自动纠正 ASR 的错字、漏字、删掉的重复；分析话轮前先听音校对。
- **字级不等于毫秒级准确。** 当前 Qwen 模型配置的 `timestamp_segment_time` 是 80 ms，SDK 还会修正不单调时间。输出小数位数不会把声学精度提高到 1 ms；短促“嗯/啊”、话轮间隔与重叠边界应人工复核。
- **声道不自动等于说话人。** 分声道保留来源，但不能消除串音、回声或同一路中的多人声音。两路人/AI 身份需要依据录音设置或听音确认。
- **拼音是文本读音，不是声学测量。** 使用整句语境、数字声调和轻声 5，`tone_sandhi=False`；多音字、人名、儿化、实际变调仍需核查。
- **中文按汉字，英文通常按词。** 不为标点生成时间；不均分英文词或数字串的时长给每个字母/数字。代码没有音素级声母韵母边界。
- **VAD 上限设为 25 秒。** 切片在 30 秒以上会停止，避免误用超长片段；VAD 最大长度处的切分仍可能打断连续讲话，需结合音频检查。

需要修改特定词组读音时，可在主程序执行前加入：

```python
from pypinyin import load_phrases_dict
load_phrases_dict({"重庆": [["chóng"], ["qìng"]]})
```
## 官方依据

- [FunASR：VAD、ASR 和采样率](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README_zh.md)
- [Qwen3-ForcedAligner 模型和接口](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Qwen 对齐器源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py)
- [Qwen 时间档位配置](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/blob/main/config.json)
- [pypinyin：整句转换和自定义读音](https://github.com/mozillazg/python-pinyin/blob/master/docs/usage.rst)
- [Praat TextGrid 格式](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html)
