# T-Stitch Jittor Engine

面向工程落地的 Jittor 复现版本，包含：
- 时序特征对齐（`TemporalAlignment`）
- Stitch 模块（门控拼接融合）
- 多尺度融合（FPN-style）
- 完整训练/验证/推理 pipeline

## 1. 目录结构

```text
t_stitch_jittor_engine/
  model.py
  dataset.py
  utils.py
  train.py
  .vscode/
    launch.json
```

## 2. 数据集组织

推荐格式：

```text
your_data/
  train/
    seq_001/
      input/
        000000.png
        000001.png
      target/
        000000.png
        000001.png
  val/
    seq_101/
      input/
      target/
```

兼容格式：
- 只有 `target`（或 `frames`）时，代码会自动合成低质输入用于训练。

## 3. 训练

```bash
cd t_stitch_jittor_engine
python train.py \
  --mode train \
  --data_root /path/to/your_data \
  --ckpt_dir ./checkpoints \
  --epochs 200 \
  --batch_size 4 \
  --num_frames 3 \
  --crop_size 256 \
  --use_cuda 1
```

恢复训练：

```bash
python train.py --mode train --data_root /path/to/your_data --resume ./checkpoints/latest.pkl
```

## 4. 推理

输入一段序列帧目录（按文件名排序）：

```bash
python train.py \
  --mode infer \
  --checkpoint ./checkpoints/best.pkl \
  --input_sequence /path/to/sequence_frames \
  --output_dir ./infer_results \
  --num_frames 3 \
  --use_cuda 1
```

## 5. 多卡与显存建议

- 多卡：使用 `mpirun` 启动，Jittor 会做梯度同步（`jt.in_mpi`）。
- 显存不足时建议：
  - 降低 `--batch_size`
  - 降低 `--crop_size`
  - 使用 `--amp_level 3~5`（设备支持时）

## 6. VSCode 调试

已提供 `.vscode/launch.json`，可直接在 VSCode 选择：
- `TStitch Train`
- `TStitch Infer`

## 7. 注意事项

- 输入尺寸需可被 4 整除（代码已做自动裁剪兜底）。
- `model.py` 里有严格维度校验，异常会直接抛出，便于定位问题。
- 训练日志与异常堆栈会输出到控制台和日志文件。

