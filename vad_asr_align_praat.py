"""VAD 切片 → ASR/拼音 → Qwen 强制对齐 → Praat TextGrid。

修改下面的输入路径和 STAGE 后运行。模型在相应步骤内加载。
"""

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import soundfile as sf
from pypinyin import Style, lazy_pinyin


BASE_DIR = Path(__file__).resolve().parent
# 只把本项目的模型放在项目目录；不修改系统 Python 或其他项目环境。
os.environ.setdefault("MODELSCOPE_CACHE", str(BASE_DIR / "models" / "funasr"))
INPUT_AUDIO = BASE_DIR / "【人AI语音交互_pilot_V2_打乱_AI BC人】CY-1029 1759-第9题-题号63-正式-a1-实际.webm"
OUTPUT_DIR = BASE_DIR / "outputs" / "CY-1029_Q63"
STAGE = "all"  # "prepare"、"vad"、"asr"、"align"、"export" 或 "all"
CHANNELS = (0, 1)  # 数组下标；输出目录使用 ch1、ch2，不猜测人/AI 身份
SAMPLE_RATE = 16000
MAX_SEGMENT_MS = 25000  # 控制在 Paraformer 常用的短音频范围内
ALIGNER_PATH = str(BASE_DIR / "models" / "Qwen3-ForcedAligner-0.6B")


def save_manifest(folder, manifest):
    (folder / "segments.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(folder):
    return json.loads((folder / "segments.json").read_text(encoding="utf-8"))


def sentence_pinyin(text):
    # 整句转换保留词组语境；非汉字逐字符返回，保证与原文本索引对应。
    readings = lazy_pinyin(
        text, style=Style.TONE3, neutral_tone_with_five=True,
        tone_sandhi=False, errors=lambda chars: list(chars),
    )
    if len(readings) != len(text):
        raise ValueError("拼音与文本长度不同，请检查该句中的特殊字符/词组读音。")
    return readings


def prepare_stereo(input_audio, folder):
    """WebM 只解码一次；保留左右声道、静音及共同的采样点时间轴。"""
    import imageio_ffmpeg

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    recording = folder / "recording.wav"
    metadata_file = folder / "recording.json"
    if recording.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata["source"] != str(Path(input_audio).resolve()):
            raise ValueError("输出目录已有其他输入的结果，请换输出目录。")
        return recording
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-n",
        "-i", str(input_audio), "-map", "0:a:0", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_f32le", str(recording),
    ], check=True)
    info = sf.info(recording)
    if info.channels != 2 or info.frames == 0:
        raise ValueError(f"要求非空双声道输入，实际声道数：{info.channels}。")
    metadata_file.write_text(json.dumps({
        "source": str(Path(input_audio).resolve()), "channels": info.channels,
        "sample_rate": info.samplerate, "num_samples": info.frames,
        "duration_s": info.frames / info.samplerate,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return recording


def vad_cut(input_audio, folder, channel=0, model=None):
    """保存全长分析音频、各 VAD 切片及它们在全长音频中的采样点边界。"""
    from funasr import AutoModel

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "segments.json").exists():
        raise FileExistsError("已有 segments.json；请换输出目录，以免覆盖人工校对。")

    audio, original_sr = sf.read(input_audio, dtype="float32", always_2d=True)
    if not 0 <= channel < audio.shape[1]:
        raise ValueError(f"输入只有 {audio.shape[1]} 个声道，CHANNEL={channel} 无效。")
    audio = audio[:, channel]
    if audio.size == 0:
        raise ValueError("输入音频为空。")

    if original_sr != SAMPLE_RATE:
        raise ValueError("请先通过 prepare_stereo 统一解码为 16 kHz。")
    sf.write(folder / "recording.wav", audio, SAMPLE_RATE, subtype="FLOAT")
    (folder / "clips").mkdir(exist_ok=True)

    vad = model if model is not None else AutoModel(
        model="fsmn-vad", device="cpu", disable_update=True,
        max_single_segment_time=MAX_SEGMENT_MS,
    )
    result = vad.generate(input=audio, fs=SAMPLE_RATE)
    spans = result[0].get("value", []) if result else []
    manifest = {
        "source": str(Path(input_audio).resolve()), "channel": channel,
        "sample_rate": SAMPLE_RATE, "num_samples": len(audio), "segments": [],
    }
    last_end = 0
    for start_ms, end_ms in spans:
        # 离线 VAD 应返回完整边界；不接受流式的 -1 占位时间。
        if start_ms < 0 or end_ms < start_ms:
            raise ValueError(f"无效 VAD 边界：{start_ms}, {end_ms}")
        first = min(len(audio), round(start_ms * SAMPLE_RATE / 1000))
        last = min(len(audio), round(end_ms * SAMPLE_RATE / 1000))
        if last == first:
            continue
        if first < last_end:
            raise ValueError("VAD 片段重叠，不能直接放到同一个 TextGrid 区间层。")
        if last - first > 30 * SAMPLE_RATE:
            raise ValueError("VAD 返回了超过 30 秒的片段，请检查最大切片长度设置。")
        segment_id = f"{len(manifest['segments']) + 1:04d}"
        wav = f"clips/{segment_id}.wav"
        sf.write(folder / wav, audio[first:last], SAMPLE_RATE, subtype="FLOAT")
        manifest["segments"].append({
            "id": segment_id, "wav": wav, "start_sample": first,
            "end_sample": last, "asr_text": "", "text": "", "pinyin": [],
            "asr_done": False,
        })
        last_end = last
    save_manifest(folder, manifest)
    print(f"VAD：已保存 {len(manifest['segments'])} 段音频。")


def asr_and_pinyin(folder, device="cpu", model=None):
    """识别已经落盘的切片，保存原始转写和可校对的 text 字段。"""
    from funasr import AutoModel

    folder = Path(folder)
    manifest = load_manifest(folder)
    if not manifest["segments"]:
        return
    asr = model if model is not None else AutoModel(
        model="paraformer-zh", punc_model="ct-punc", device=device,
        disable_update=True, trust_remote_code=False,
    )
    for segment in manifest["segments"]:
        # 逐段保存，重跑时保留已有转写及人工校对；空识别也有完成标记。
        if segment.get("asr_done") or segment["asr_text"] or segment["text"]:
            continue
        result = asr.generate(input=str(folder / segment["wav"]))
        text = result[0].get("text", "").strip() if result else ""
        segment["asr_text"] = text
        segment["text"] = text
        segment["pinyin"] = sentence_pinyin(text)
        segment["asr_done"] = True
        save_manifest(folder, manifest)
        print(f"ASR {segment['id']}：{text or '[空转写，需复核]'}")
    save_manifest(folder, manifest)


def map_pinyin(text, aligned_items):
    """整句拼音按文本位置映射到 Qwen 返回的汉字/英文词，不按字重新猜读音。"""
    readings = sentence_pinyin(text)
    # Qwen 的文本处理保留字母、数字及英文单引号，忽略标点和空白。
    pairs = [
        (char, reading) for char, reading in zip(text, readings)
        if char == "'" or unicodedata.category(char)[0] in "LN"
    ]
    actual = "".join(item.text for item in aligned_items)
    expected = "".join(char for char, _ in pairs)
    if actual != expected:
        raise ValueError(f"对齐文本与输入不同：{actual!r} != {expected!r}")
    labels = []
    cursor = 0
    for item in aligned_items:
        part = pairs[cursor:cursor + len(item.text)]
        # 中文单元通常是一个汉字；英文/数字保留原标签，不假造音素边界。
        labels.append(" ".join(p for _, p in part) if len(part) == 1 else item.text)
        cursor += len(item.text)
    return labels


def interval_partition(spans, duration):
    """填充空白区间，使一个 IntervalTier 连续覆盖 0 到全长。"""
    intervals = []
    cursor = 0.0
    for start, end, label in spans:
        if not (math.isfinite(start) and math.isfinite(end)
                and cursor <= start < end <= duration):
            raise ValueError(f"不能写入 TextGrid 的区间：{start}, {end}, {label!r}")
        if start > cursor:
            intervals.append((cursor, start, ""))
        intervals.append((start, end, label))
        cursor = end
    if cursor < duration:
        intervals.append((cursor, duration, ""))
    return intervals


def write_textgrid(path, duration, tiers, review_points=None, point_tiers=None):
    """直接写标准长文本 TextGrid；无须额外的 TextGrid 第三方库。"""
    def quoted(value):
        # Praat 使用两个双引号表示标签中的一个双引号。
        return '"' + str(value).replace('"', '""').replace("\n", " ").replace("\r", " ") + '"'

    if point_tiers is None:
        point_tiers = {"review": review_points or []}
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0", f"xmax = {duration:.17g}", "tiers? <exists>",
        f"size = {len(tiers) + len(point_tiers)}", "item []:",
    ]
    for tier_index, (name, spans) in enumerate(tiers.items(), start=1):
        intervals = interval_partition(spans, duration)
        lines.extend([
            f"    item [{tier_index}]:", '        class = "IntervalTier"',
            f"        name = {quoted(name)}", "        xmin = 0",
            f"        xmax = {duration:.17g}", f"        intervals: size = {len(intervals)}",
        ])
        for index, (start, end, label) in enumerate(intervals, start=1):
            lines.extend([
                f"        intervals [{index}]:", f"            xmin = {start:.17g}",
                f"            xmax = {end:.17g}", f"            text = {quoted(label)}",
            ])
    # 多个异常可落在同一时刻；合并成一个点，满足 TextTier 时间严格递增的要求。
    for tier_index, (name, review) in enumerate(point_tiers.items(), len(tiers) + 1):
        points = {}
        for time, label in review:
            if not math.isfinite(time) or not 0 <= time <= duration:
                raise ValueError(f"复核点越界：{time}")
            points.setdefault(time, []).append(label)
        lines.extend([
            f"    item [{tier_index}]:", '        class = "TextTier"',
            f"        name = {quoted(name)}", "        xmin = 0", f"        xmax = {duration:.17g}",
            f"        points: size = {len(points)}",
        ])
        for index, (time, labels) in enumerate(sorted(points.items()), start=1):
            lines.extend([
                f"        points [{index}]:", f"            number = {time:.17g}",
                f"            mark = {quoted(' | '.join(labels))}",
            ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def align_and_export(folder, device="cpu", dtype=None, model=None):
    """对齐 text，输出切片/全长 TextGrid，以及包含全部原始时间的 CSV。"""
    from qwen_asr import Qwen3ForcedAligner

    folder = Path(folder)
    manifest = load_manifest(folder)
    sr = manifest["sample_rate"]
    duration = manifest["num_samples"] / sr
    whole_tiers = {"segments": [], "hanzi": [], "pinyin": []}
    whole_review, rows = [], []
    for segment in manifest["segments"]:
        offset = segment["start_sample"] / sr
        segment_end = segment["end_sample"] / sr
        clip_duration = (segment["end_sample"] - segment["start_sample"]) / sr

        def global_time(local_time):
            # 片段端点直接使用采样点算出的时间，消除浮点相加的边界误差。
            if local_time == 0:
                return offset
            if local_time == clip_duration:
                return segment_end
            return offset + local_time

        text = segment["text"].strip()
        segment["pinyin"] = sentence_pinyin(text)  # 校对 text 后重新计算拼音
        whole_tiers["segments"].append((offset, segment_end, text))
        clip_tiers = {"segments": [(0.0, clip_duration, text)], "hanzi": [], "pinyin": []}
        clip_review = []
        if not any(unicodedata.category(c)[0] in "LN" for c in text):
            clip_review.append((0.0, "empty_transcript：无可对齐文本，请复核本段"))
        else:
            if model is None:
                model = Qwen3ForcedAligner.from_pretrained(
                    ALIGNER_PATH, device_map=device, dtype=dtype,
                )
            items = list(model.align(
                audio=str(folder / segment["wav"]), text=text, language="Chinese",
            )[0])
            readings = map_pinyin(text, items)
            last_end = 0.0
            for index, (item, reading) in enumerate(zip(items, readings), start=1):
                start, end = float(item.start_time), float(item.end_time)
                # 零时长、越界、逆序或重叠不能作为 Praat 的正常区间写出。
                # 保留原始时间到 CSV，并在 review 点层提示；不伪造时长。
                problems = []
                if not (math.isfinite(start) and math.isfinite(end)):
                    problems.append("nonfinite")
                else:
                    if end <= start:
                        problems.append("zero_duration" if end == start else "reversed")
                    if start < 0 or end > clip_duration:
                        problems.append("outside_clip")
                    if start < last_end:
                        problems.append("overlap")
                status = ";".join(problems) or "ok"
                rows.append({
                    "channel": manifest["channel"] + 1,
                    "segment_id": segment["id"], "unit_index": index,
                    "text": item.text, "pinyin": reading,
                    "start_local_s": start, "end_local_s": end,
                    "start_global_s": global_time(start), "end_global_s": global_time(end),
                    "status": status,
                })
                if problems:
                    at = min(max(start, 0.0), clip_duration) if math.isfinite(start) else 0.0
                    clip_review.append((at, f"{index}:{item.text}/{reading} [{status}]"))
                    continue
                clip_tiers["hanzi"].append((start, end, item.text))
                clip_tiers["pinyin"].append((start, end, reading))
                whole_tiers["hanzi"].append((global_time(start), global_time(end), item.text))
                whole_tiers["pinyin"].append((global_time(start), global_time(end), reading))
                last_end = end
        write_textgrid((folder / segment["wav"]).with_suffix(".TextGrid"),
                       clip_duration, clip_tiers, clip_review)
        whole_review.extend((global_time(time), f"{segment['id']} {label}")
                            for time, label in clip_review)
        print(f"对齐 {segment['id']}：{len(clip_tiers['hanzi'])} 个有效区间，"
              f"{len(clip_review)} 个复核点。")

    write_units(folder / "units.csv", rows)
    save_manifest(folder, manifest)
    write_textgrid(folder / "recording.TextGrid", duration, whole_tiers, whole_review)
    print(f"Praat 文件已导出到：{folder.resolve()}")
    return duration, whole_tiers, whole_review, rows


def write_units(path, rows):
    fields = ["channel", "segment_id", "unit_index", "text", "pinyin", "start_local_s",
              "end_local_s", "start_global_s", "end_global_s", "status", "row_type"]
    # 同步旧 CSV 时它本身也是输入；写完临时文件再替换，避免中断丢失原始字行。
    path = Path(path)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        with pending.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"row_type": "unit", **row})
        pending.replace(path)


def gap_label(start, end):
    # 只在判定时舍去浮点尾差，保留 Praat 的原始边界。
    seconds = round(end - start, 9)
    if seconds < 0.080:
        return ""
    return "sli" if seconds <= 0.200 else "ps"


def export_sli_ps(source, hanzi_gaps=None):
    """另存带标签的 TextGrid；可同时取出汉字层的同一批空白供 CSV 使用。"""
    import parselmouth
    from parselmouth.praat import call

    source = Path(source)
    grid = parselmouth.read(str(source))
    for tier in range(1, call(grid, "Get number of tiers") + 1):
        if not call(grid, "Is interval tier", tier):
            continue
        name = call(grid, "Get tier name", tier)
        collect = hanzi_gaps is not None and (name == "hanzi" or name.endswith("_hanzi"))
        if collect:
            hanzi_gaps[name] = []
        for index in range(1, call(grid, "Get number of intervals", tier) + 1):
            if call(grid, "Get label of interval", tier, index).strip():
                continue
            start = call(grid, "Get start time of interval", tier, index)
            end = call(grid, "Get end time of interval", tier, index)
            label = gap_label(start, end)
            call(grid, "Set interval text", tier, index, label)
            if collect:
                hanzi_gaps[name].append((start, end, label))
    target = source.with_stem(source.stem + "_sli_ps")
    grid.save_as_text_file(str(target))
    return target


def gap_rows(gaps, channel, manifest):
    """全长汉字层的每个空白只记一行；跨切片的区间只使用全局时间。"""
    sr = manifest["sample_rate"]
    segments = iter(manifest["segments"])
    segment = next(segments, None)
    for start, end, label in gaps:
        while segment and segment["end_sample"] / sr <= start:
            segment = next(segments, None)
        row = {"row_type": "gap", "channel": channel, "text": label, "pinyin": label,
               "start_global_s": start, "end_global_s": end, "status": "gap"}
        if segment and segment["start_sample"] / sr <= start and end <= segment["end_sample"] / sr:
            offset = segment["start_sample"] / sr
            row.update(segment_id=segment["id"], start_local_s=start - offset, end_local_s=end - offset)
        yield row


def sync_units(path, gaps):
    with path.open(encoding="utf-8-sig", newline="") as f:
        # 兼容只有字行的旧 CSV；重复导出时按当前 TextGrid 重建所有空白行。
        rows = [r for r in csv.DictReader(f) if r.get("row_type") != "gap"]
    rows.extend(gaps)
    rows.sort(key=lambda r: (float(r["start_global_s"]), int(r["channel"])))
    write_units(path, rows)


def export_annotations(folder):
    """从已有原始 TextGrid、切片清单和 units.csv 同步导出，无需再次推理。"""
    folder = Path(folder)
    combined_gaps = {}
    export_sli_ps(folder / "recording.TextGrid", combined_gaps)
    combined_rows = []
    for index in CHANNELS:
        channel = index + 1
        channel_dir = folder / f"ch{channel}"
        manifest = load_manifest(channel_dir)
        for segment in manifest["segments"]:
            export_sli_ps((channel_dir / segment["wav"]).with_suffix(".TextGrid"))
        local_gaps = {}
        export_sli_ps(channel_dir / "recording.TextGrid", local_gaps)
        sync_units(channel_dir / "units.csv", gap_rows(local_gaps["hanzi"], channel, manifest))
        combined_rows.extend(gap_rows(combined_gaps[f"ch{channel}_hanzi"], channel, manifest))
    sync_units(folder / "units.csv", combined_rows)
    print(f"已同步 *_sli_ps.TextGrid 和 units.csv：{folder}")


def main(stage=None, folder=None, audio=None):
    stage = stage or STAGE
    output_dir = Path(folder or OUTPUT_DIR).resolve()
    if stage == "export":
        export_annotations(output_dir)
        return
    if stage not in {"prepare", "vad", "asr", "align", "all"}:
        raise ValueError(f"未知阶段：{stage}")
    folders = [(channel, output_dir / f"ch{channel + 1}") for channel in CHANNELS]
    if stage in {"prepare", "vad", "all"}:
        recording = prepare_stereo(audio or INPUT_AUDIO, output_dir)
    if stage == "prepare":
        return

    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
             ) if device.startswith("cuda") else torch.float32
    if stage in {"vad", "all"}:
        vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True,
                        max_single_segment_time=MAX_SEGMENT_MS)
        for channel, folder in folders:
            if (folder / "segments.json").exists():
                previous = load_manifest(folder)
                if previous["channel"] != channel or previous["source"] != str(recording.resolve()):
                    raise ValueError("已有切片不属于当前音频/声道，请换输出目录。")
                print(f"{folder.name}：复用已有 VAD 切片。")
                continue
            vad_cut(recording, folder, channel, vad)
        del vad
    if stage in {"asr", "all"}:
        asr = AutoModel(model="paraformer-zh", punc_model="ct-punc", device=device,
                        disable_update=True, trust_remote_code=False)
        for _, folder in folders:
            asr_and_pinyin(folder, device, asr)
        del asr
    if stage in {"align", "all"}:
        from qwen_asr import Qwen3ForcedAligner

        aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER_PATH, device_map=device, dtype=dtype)
        tiers, reviews, rows = {}, {}, []
        duration = sf.info(output_dir / "recording.wav").duration
        for _, folder in folders:
            channel_duration, channel_tiers, points, channel_rows = align_and_export(
                folder, device, dtype, aligner,
            )
            if channel_duration != duration:
                raise ValueError("两个声道必须使用同一全长时间轴。")
            tiers.update({f"{folder.name}_{name}": spans for name, spans in channel_tiers.items()})
            reviews[f"{folder.name}_review"] = points
            rows.extend(channel_rows)
        # 两个声道各有独立区间层，所以跨声道重叠不会被删除或合并。
        write_textgrid(output_dir / "recording.TextGrid", duration, tiers, point_tiers=reviews)
        write_units(output_dir / "units.csv", sorted(rows, key=lambda r: (
            r["start_global_s"], r["channel"], r["unit_index"])))
        export_annotations(output_dir)


def cli(default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, help="分析结果目录")
    parser.add_argument("--stage", choices=("prepare", "vad", "asr", "align", "export", "all"))
    parser.add_argument("--audio", type=Path, help="从新的双声道录音开始（需使用空目录）")
    args = parser.parse_args()
    folder = args.folder or OUTPUT_DIR
    if args.audio:
        if not args.audio.is_file():
            parser.error(f"找不到音频：{args.audio}")
        if folder.exists() and any(folder.iterdir()):
            parser.error("处理新的音频时，请指定新的或空的输出目录。")
        if args.stage not in (None, "all", "prepare", "vad"):
            parser.error("--audio 适用于 all、prepare 或 vad 阶段。")
    stage = args.stage or ("all" if args.audio else default_stage or STAGE)
    main(stage, folder, args.audio)


if __name__ == "__main__":
    cli()
