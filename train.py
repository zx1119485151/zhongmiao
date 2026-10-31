#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys


MODEL_YAML = "model/yolov8n_SCS.yaml"
PRETRAINED_WEIGHTS = "pre_weights/yolov8n.pt"
DATA_YAML = "dataset/dataset.yaml"  

EPOCHS = 300
IMGSZ = 1280  
BATCH = 16
PATIENCE = 100
OPTIMIZER = "SGD"  
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
COS_LR = False
SEED = 0
AMP = True
MULTI_SCALE = False
DEVICE = "0"  
WORKERS = 8
PROJECT = "ex/work/runs"
NAME = None  # None -> 用模型 YAML 的文件名作为实验名

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def require_file(parser, value, option, suffixes):
    if not value or not value.strip():
        parser.error(f"请填写 {option}：可修改脚本顶部配置，或通过命令行传入。")
    path = project_path(value)
    if path.suffix.lower() not in suffixes:
        parser.error(f"{option} 的扩展名必须是 {', '.join(suffixes)}：{path}")
    if not path.is_file():
        parser.error(f"{option} 文件不存在：{path}")
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用指定模型 YAML + 指定预训练 .pt 权重训练（不是断点续训）。"
    )
    parser.add_argument("--model-yaml", default=MODEL_YAML, help="模型结构 YAML 路径")
    parser.add_argument("--weights", default=PRETRAINED_WEIGHTS, help="预训练 .pt 权重路径")
    parser.add_argument("--data", default=DATA_YAML, help="数据集 YAML 路径")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="YOLOv8: 1280；train10: 1024，需手动指定")
    parser.add_argument("--batch", type=int, default=BATCH, help="正整数，或 -1 自动估算")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值，0 表示禁用早停")
    parser.add_argument("--optimizer", default=OPTIMIZER, help="例如 auto、SGD、AdamW")
    parser.add_argument("--lr0", type=float, default=LR0)
    parser.add_argument("--lrf", type=float, default=LRF)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=float, default=WARMUP_EPOCHS)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=COS_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=AMP)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=MULTI_SCALE,
                        help="是否启用多尺度；默认关闭")
    parser.add_argument("--device", default=DEVICE, help="例如 0、0,1、cpu")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--project", default=PROJECT, help="输出目录，相对项目根目录")
    parser.add_argument("--name", default=NAME, help="实验名称，默认取模型 YAML 文件名；重复名称会自动递增")
    parser.add_argument("--task", choices=["detect", "segment", "pose", "obb"], default=None,
                        help="默认从 YAML 推断；自定义 YAML 无法推断时手动指定")
    parser.add_argument("--check", action="store_true", help="仅检查路径并打印配置，不加载模型、不训练")
    args = parser.parse_args()
    args.model_yaml = require_file(parser, args.model_yaml, "--model-yaml", (".yaml", ".yml"))
    args.weights = require_file(parser, args.weights, "--weights", (".pt",))
    args.data = require_file(parser, args.data, "--data", (".yaml", ".yml"))
    if args.epochs <= 0 or args.imgsz <= 0:
        parser.error("--epochs 和 --imgsz 必须大于 0")
    if args.batch != -1 and args.batch <= 0:
        parser.error("--batch 必须是正整数或 -1")
    if args.workers < 0 or args.patience < 0 or args.seed < 0:
        parser.error("--workers、--patience 和 --seed 不能小于 0")
    if not args.lr0 > 0 or not 0 <= args.lrf <= 1 or not 0 <= args.momentum <= 1:
        parser.error("--lr0 必须大于 0；--lrf 和 --momentum 必须在 [0, 1] 内")
    if not args.weight_decay >= 0 or not args.warmup_epochs >= 0:
        parser.error("--weight-decay 和 --warmup-epochs 不能小于 0")
    args.name = args.name or Path(args.model_yaml).stem
    if not args.project.strip() or not args.name.strip():
        parser.error("--project 和 --name 不能为空")
    args.project = str(project_path(args.project))
    return args


def main():
    args = parse_args()
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "seed": args.seed,
        "amp": args.amp,
        "multi_scale": 0.25 if args.multi_scale else 0.0,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": False,
        "pretrained": args.weights,
        "resume": False,
    }
    print(f"项目根目录：{ROOT}")
    print(f"模型 YAML：{args.model_yaml}")
    print(f"预训练权重：{args.weights}")
    print("训练配置（multi_scale=0.0 表示关闭）：")
    for key, value in train_args.items():
        print(f"  {key}={value}")
    if args.optimizer.lower() == "auto":
        print("注意：optimizer=auto 时，Ultralytics 会自动选择优化器，并忽略配置的 lr0 和 momentum。")
    if args.check:
        print("路径检查通过；未加载模型，也未启动训练。YAML 内容和权重兼容性需实际加载时验证。")
        return

 
    os.chdir(ROOT)
    from ultralytics import YOLO


    model = YOLO(args.model_yaml, task=args.task)
    model.load(args.weights)
    model.train(**train_args)
    print(f"训练结束，结果保存在：{model.trainer.save_dir}")


if __name__ == "__main__":
    main()
