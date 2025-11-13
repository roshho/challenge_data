#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from ultralytics import YOLO


logger = logging.getLogger(__name__)


def gpu_info() -> str:
	if not torch.cuda.is_available():
		return "No GPU detected."
	idx = 0
	name = torch.cuda.get_device_name(idx)
	mem = torch.cuda.get_device_properties(idx).total_memory / 1e9
	return f"GPU: {name} ({mem:.1f} GB)"


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train YOLO tip detector")
	p.add_argument("--data", type=str, default="data.yaml", help="Path to data YAML")
	p.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained model to load")
	p.add_argument("--epochs", type=int, default=100)
	p.add_argument("--imgsz", type=int, default=640)
	p.add_argument("--batch", type=int, default=16)
	p.add_argument("--project", type=str, default="runs/train")
	p.add_argument("--name", type=str, default="tip_detector")
	p.add_argument("--optimizer", type=str, default="SGD", help="Optimizer to use (e.g., SGD, Adam)")
	p.add_argument("--save-period", type=int, default=10, help="Checkpoint save period (epochs)")
	p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
	return p.parse_args()


def train_yolo(
	data: str,
	model_path: str = "yolov8s.pt",
	epochs: int = 100,
	imgsz: int = 640,
	batch: int = 16,
	project: str = "runs/train",
	name: str = "tip_detector",
	optimizer: str = "SGD",
	save_period: int = 10,
	patience: int = 20,
) -> Tuple[YOLO, Optional[Any]]:
	logger.info("Starting YOLO training")
	logger.info(gpu_info())

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	logger.info("Device: %s", device)

	model = YOLO(model_path)

	try:
		results = model.train(
			data=data,
			epochs=epochs,
			imgsz=imgsz,
			batch=batch,
			optimizer=optimizer,
			device=device,
			project=project,
			name=name,
			patience=patience,
			save=True,
			save_period=save_period,
			plots=True,
			val=True,
			verbose=True,
		)
	except Exception as exc:  # keep broad to surface failures during training
		logger.exception("Training failed: %s", exc)
		return model, None

	# Best-effort metrics extraction (Ultralytics API can change)
	metrics = getattr(results, "results_dict", None)
	if isinstance(metrics, dict):
		map50 = metrics.get("metrics/mAP50(B)", "N/A")
		map50_95 = metrics.get("metrics/mAP50-95(B)", "N/A")
	else:
		map50 = map50_95 = "N/A"

	logger.info("Training finished")
	logger.info("Best model saved to: %s", Path(project) / name / "weights" / "best.pt")
	logger.info("Results folder: %s", Path(project) / name)
	logger.info("mAP@0.5: %s", map50)
	logger.info("mAP@0.5:0.95: %s", map50_95)

	return model, results


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	args = parse_args()
	train_yolo(
		data=args.data,
		model_path=args.model,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		project=args.project,
		name=args.name,
		optimizer=args.optimizer,
		save_period=args.save_period,
		patience=args.patience,
	)


if __name__ == "__main__":
	main()
