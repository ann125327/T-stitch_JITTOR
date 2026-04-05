import argparse
import os
import shutil
import traceback

# 避免某些环境下 Jittor cache 无写权限
if "JITTOR_HOME" not in os.environ:
    _local_home = os.path.join(os.path.dirname(__file__), ".jittor_home")
    os.environ["JITTOR_HOME"] = _local_home
    os.environ.setdefault("HOME", _local_home)
    os.environ.setdefault("USERPROFILE", _local_home)

import numpy as np
import jittor as jt

from dataset import VideoSequenceDataset, read_rgb, to_chw
from model import TStitchNet
from utils import (
    AverageMeter,
    TStitchLoss,
    compute_psnr,
    compute_ssim_fast,
    dump_json,
    ensure_dir,
    load_checkpoint,
    safe_float,
    save_checkpoint,
    save_image,
    set_random_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser("T-Stitch Jittor Training/Inference")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--input_sequence", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--synthetic_degrade", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=8.0)

    parser.add_argument("--w_rec", type=float, default=1.0)
    parser.add_argument("--w_edge", type=float, default=0.05)
    parser.add_argument("--w_temporal", type=float, default=0.1)
    parser.add_argument("--w_pyramid", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--amp_level", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    return parser.parse_args()


def configure_runtime(args, logger):
    set_random_seed(args.seed)

    if hasattr(jt.flags, "use_cuda"):
        jt.flags.use_cuda = int(args.use_cuda)
    if hasattr(jt.flags, "auto_mixed_precision_level"):
        jt.flags.auto_mixed_precision_level = int(args.amp_level)

    logger.info(
        "Runtime | use_cuda=%s amp_level=%s in_mpi=%s",
        getattr(jt.flags, "use_cuda", "NA"),
        getattr(jt.flags, "auto_mixed_precision_level", "NA"),
        jt.in_mpi,
    )


def build_model(args):
    return TStitchNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        num_frames=args.num_frames,
    )


def train_one_epoch(model, loader, optimizer, criterion, epoch, logger, args):
    model.train()
    loss_meter = AverageMeter()
    rec_meter = AverageMeter()
    edge_meter = AverageMeter()
    temp_meter = AverageMeter()

    for step, batch in enumerate(loader):
        if args.max_steps_per_epoch > 0 and step >= args.max_steps_per_epoch:
            break

        lq, gt = batch
        if len(lq.shape) != 5:
            raise ValueError(f"Train input shape error: expect [B,T,C,H,W], got {lq.shape}")
        if len(gt.shape) != 4:
            raise ValueError(f"Train target shape error: expect [B,C,H,W], got {gt.shape}")

        try:
            outputs = model(lq)
            losses = criterion(outputs, gt)
            total_loss = losses["total"]

            optimizer.zero_grad()
            optimizer.backward(total_loss)
            if args.grad_clip > 0:
                optimizer.clip_grad_norm(args.grad_clip, 2)
            optimizer.step()
        except Exception:
            logger.error("Train step failed at epoch=%d step=%d", epoch, step)
            logger.error(traceback.format_exc())
            raise

        bs = int(lq.shape[0])
        loss_meter.update(safe_float(total_loss), bs)
        rec_meter.update(safe_float(losses["rec"]), bs)
        edge_meter.update(safe_float(losses["edge"]), bs)
        temp_meter.update(safe_float(losses["temporal"]), bs)

        if step % args.print_freq == 0:
            logger.info(
                "Epoch[%d] Step[%d/%d] total=%.6f rec=%.6f edge=%.6f temporal=%.6f",
                epoch,
                step,
                len(loader),
                loss_meter.avg,
                rec_meter.avg,
                edge_meter.avg,
                temp_meter.avg,
            )

    return {
        "loss": loss_meter.avg,
        "rec": rec_meter.avg,
        "edge": edge_meter.avg,
        "temporal": temp_meter.avg,
    }


def validate(model, loader, criterion, logger):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with jt.no_grad():
        for batch in loader:
            lq, gt = batch
            outputs = model(lq)
            losses = criterion(outputs, gt)
            pred = outputs["pred"]

            bs = int(lq.shape[0])
            loss_meter.update(safe_float(losses["total"]), bs)
            psnr_meter.update(compute_psnr(pred, gt), bs)
            ssim_meter.update(compute_ssim_fast(pred, gt), bs)

    logger.info("Validation | loss=%.6f psnr=%.4f ssim=%.4f", loss_meter.avg, psnr_meter.avg, ssim_meter.avg)
    return {
        "loss": loss_meter.avg,
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
    }


def run_train(args):
    ensure_dir(args.ckpt_dir)
    logger = setup_logger(log_dir=os.path.join(args.ckpt_dir, "logs"), name="train")
    configure_runtime(args, logger)

    model = build_model(args)
    criterion = TStitchLoss(args.w_rec, args.w_edge, args.w_temporal, args.w_pyramid)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)

    train_loader = VideoSequenceDataset(
        data_root=args.data_root,
        split="train",
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        synthetic_degrade=bool(args.synthetic_degrade),
        noise_std=args.noise_std,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = None
    try:
        val_loader = VideoSequenceDataset(
            data_root=args.data_root,
            split="val",
            num_frames=args.num_frames,
            crop_size=0,
            synthetic_degrade=False,
            noise_std=args.noise_std,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
        )
    except Exception as e:
        logger.warning("Validation set not found or invalid, skip validation. reason=%s", str(e))

    start_epoch = 0
    best_psnr = -1e9
    if args.resume:
        start_epoch, best_psnr, sched_state = load_checkpoint(args.resume, model, optimizer)
        if isinstance(sched_state, dict) and "last_epoch" in sched_state and hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = int(sched_state["last_epoch"])
        logger.info("Resume from %s | start_epoch=%d best_psnr=%.4f", args.resume, start_epoch, best_psnr)
    else:
        # Save an initial checkpoint so interrupted runs still have a valid model file.
        init_sched_state = {"last_epoch": int(getattr(scheduler, "last_epoch", -1))}
        init_path = os.path.join(args.ckpt_dir, "latest.pkl")
        save_checkpoint(init_path, model, optimizer, 0, best_psnr, init_sched_state)
        logger.info("Saved initial checkpoint: %s", init_path)

    cfg_path = os.path.join(args.ckpt_dir, "train_config.json")
    dump_json(vars(args), cfg_path)

    rank = int(getattr(jt, "rank", 0)) if jt.in_mpi else 0
    is_rank0 = (not jt.in_mpi) or (rank == 0)

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, epoch, logger, args)

        if val_loader is not None:
            val_stats = validate(model, val_loader, criterion, logger)
            metric = val_stats["psnr"]
        else:
            val_stats = {}
            metric = -train_stats["loss"]

        scheduler.step()

        if is_rank0 and ((epoch + 1) % args.save_freq == 0):
            sched_state = {"last_epoch": int(getattr(scheduler, "last_epoch", epoch))}
            latest_path = os.path.join(args.ckpt_dir, "latest.pkl")
            save_checkpoint(latest_path, model, optimizer, epoch + 1, best_psnr, sched_state)

            if metric > best_psnr:
                best_psnr = metric
                best_path = os.path.join(args.ckpt_dir, "best.pkl")
                save_checkpoint(best_path, model, optimizer, epoch + 1, best_psnr, sched_state)
                logger.info("New best checkpoint saved: %s (metric=%.4f)", best_path, best_psnr)

        logger.info(
            "Epoch[%d/%d] done | train_loss=%.6f best_metric=%.4f",
            epoch + 1,
            args.epochs,
            train_stats["loss"],
            best_psnr,
        )


def _collect_sequence_images(seq_dir: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [os.path.join(seq_dir, x) for x in os.listdir(seq_dir) if x.lower().endswith(exts)]
    return sorted(files)


def _crop_to_divisible(frames: list, div: int = 4):
    h = min(x.shape[0] for x in frames)
    w = min(x.shape[1] for x in frames)
    h = h - (h % div)
    w = w - (w % div)
    cropped = [x[:h, :w, :] for x in frames]
    return cropped


def run_infer(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for infer mode.")
    if not args.input_sequence or not os.path.isdir(args.input_sequence):
        raise ValueError("--input_sequence must be a valid image folder.")

    logger = setup_logger(log_dir=os.path.join(args.output_dir, "logs"), name="infer")
    configure_runtime(args, logger)
    ensure_dir(args.output_dir)

    model = build_model(args)
    load_checkpoint(args.checkpoint, model, optimizer=None)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    frame_paths = _collect_sequence_images(args.input_sequence)
    if len(frame_paths) < args.num_frames:
        raise RuntimeError(f"Need >= {args.num_frames} frames, got {len(frame_paths)}")

    radius = args.num_frames // 2
    # 边界帧直接拷贝
    for i in range(radius):
        shutil.copy2(frame_paths[i], os.path.join(args.output_dir, os.path.basename(frame_paths[i])))
        shutil.copy2(frame_paths[-1 - i], os.path.join(args.output_dir, os.path.basename(frame_paths[-1 - i])))

    with jt.no_grad():
        for i in range(radius, len(frame_paths) - radius):
            window = frame_paths[i - radius:i + radius + 1]
            frames = [read_rgb(p) for p in window]
            frames = _crop_to_divisible(frames, 4)
            arr = np.stack([to_chw(x) for x in frames], axis=0).astype(np.float32)  # [T,C,H,W]
            inp = jt.array(arr).unsqueeze(0)  # [1,T,C,H,W]

            out = model(inp)["pred"][0]  # [C,H,W]
            save_path = os.path.join(args.output_dir, os.path.basename(frame_paths[i]))
            save_image(out, save_path)

            if (i - radius) % 20 == 0:
                logger.info("Infer progress: %d/%d", i - radius + 1, len(frame_paths) - 2 * radius)

    logger.info("Inference done. Results saved to %s", args.output_dir)


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise
