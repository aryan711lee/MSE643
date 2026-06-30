from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
	path: Path
	label: int
	structure: str
	group_id: str


@dataclass
class TrainConfig:
	data_dir: Path
	image_size: int = 224
	batch_size: int = 32
	epochs: int = 15
	learning_rate: float = 1e-4
	num_workers: int = 2
	seed: int = 42
	model_name: str = "resnet18"
	use_class_weights: bool = True
	device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
	return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_label(path: Path) -> int:
	parts = {part.lower() for part in path.parts}
	if "cracked" in parts or "crack" in parts:
		return 1
	if "non-cracked" in parts or "noncracked" in parts or "non_cracked" in parts:
		return 0
	raise ValueError(f"Could not infer label from path: {path}")


def infer_structure(path: Path) -> str:
	for part in path.parts:
		upper = part.upper()
		if upper in {"D", "W", "P"}:
			return upper
	return "unknown"


def infer_group_id(path: Path) -> str:
	structure = infer_structure(path)
	stem = path.stem
	prefix = stem.split("_")[0]
	prefix = re.sub(r"[^A-Za-z0-9]+", "", prefix)
	return f"{structure}_{prefix or stem}"


def discover_dataset(data_dir: Path) -> List[Sample]:
	if not data_dir.exists():
		raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

	samples: List[Sample] = []
	for path in sorted(data_dir.rglob("*")):
		if not path.is_file() or not is_image_file(path):
			continue
		samples.append(
			Sample(
				path=path,
				label=infer_label(path),
				structure=infer_structure(path),
				group_id=infer_group_id(path),
			)
		)

	if not samples:
		raise ValueError(f"No image files found under {data_dir}")
	return samples


def summarize_dataset(samples: Sequence[Sample]) -> Dict[str, Dict[str, int]]:
	summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"cracked": 0, "non_cracked": 0})
	for sample in samples:
		key = sample.structure
		if sample.label == 1:
			summary[key]["cracked"] += 1
		else:
			summary[key]["non_cracked"] += 1
	return dict(summary)


def print_dataset_summary(samples: Sequence[Sample]) -> None:
	summary = summarize_dataset(samples)
	total = len(samples)
	cracked = sum(sample.label for sample in samples)
	print(f"Total images: {total}")
	print(f"Cracked: {cracked}")
	print(f"Non-cracked: {total - cracked}")
	for structure in sorted(summary):
		counts = summary[structure]
		print(f"{structure}: cracked={counts['cracked']} non_cracked={counts['non_cracked']}")


def group_stratified_split(
	samples: Sequence[Sample],
	train_size: float = 0.7,
	val_size: float = 0.15,
	test_size: float = 0.15,
	seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
	if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=0.0, abs_tol=1e-6):
		raise ValueError("train_size + val_size + test_size must sum to 1.0")

	groups = np.array([sample.group_id for sample in samples])
	labels = np.array([sample.label for sample in samples])
	indices = np.arange(len(samples))

	unique_groups = np.unique(groups)
	group_label = []
	for group in unique_groups:
		group_indices = indices[groups == group]
		group_label.append(int(labels[group_indices].mean() >= 0.5))

	train_groups, temp_groups = train_test_split(
		unique_groups,
		test_size=(1.0 - train_size),
		random_state=seed,
		stratify=group_label,
	)

	temp_mask = np.isin(unique_groups, temp_groups)
	temp_group_labels = [group_label[i] for i, flag in enumerate(temp_mask) if flag]
	val_relative = val_size / (val_size + test_size)
	val_groups, test_groups = train_test_split(
		temp_groups,
		test_size=(1.0 - val_relative),
		random_state=seed,
		stratify=temp_group_labels,
	)

	def collect(selected_groups: Iterable[str]) -> List[int]:
		selected = set(selected_groups)
		return [i for i, sample in enumerate(samples) if sample.group_id in selected]

	return collect(train_groups), collect(val_groups), collect(test_groups)


class SDNET2018Dataset(Dataset):
	def __init__(self, samples: Sequence[Sample], transform=None):
		self.samples = list(samples)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		sample = self.samples[index]
		image = Image.open(sample.path).convert("RGB")
		if self.transform is not None:
			image = self.transform(image)
		return image, sample.label, str(sample.path)


def build_transforms(image_size: int):
	train_transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomRotation(10),
			transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	eval_transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return train_transform, eval_transform


class BaselineCNN(nn.Module):
	def __init__(self, num_classes: int = 2):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.3),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(128, num_classes),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		features = self.features(inputs)
		return self.classifier(features)


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
	if model_name == "baseline_cnn":
		return BaselineCNN(num_classes=num_classes)

	if model_name == "resnet18":
		weights = models.ResNet18_Weights.IMAGENET1K_V1
		model = models.resnet18(weights=weights)
		for parameter in model.parameters():
			parameter.requires_grad = False
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		for parameter in model.fc.parameters():
			parameter.requires_grad = True
		return model

	if model_name == "resnet50":
		weights = models.ResNet50_Weights.IMAGENET1K_V2
		model = models.resnet50(weights=weights)
		for parameter in model.parameters():
			parameter.requires_grad = False
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		for parameter in model.fc.parameters():
			parameter.requires_grad = True
		return model

	raise ValueError(f"Unsupported model_name: {model_name}")


def compute_class_weights(samples: Sequence[Sample]) -> torch.Tensor:
	counts = Counter(sample.label for sample in samples)
	total = sum(counts.values())
	weights = [total / (2.0 * counts.get(label, 1)) for label in range(2)]
	return torch.tensor(weights, dtype=torch.float32)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)


def compute_metrics(targets: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]) -> Dict[str, float]:
	metrics: Dict[str, float] = {
		"accuracy": accuracy_score(targets, predictions),
		"precision": precision_score(targets, predictions, zero_division=0),
		"recall": recall_score(targets, predictions, zero_division=0),
		"f1": f1_score(targets, predictions, zero_division=0),
	}
	try:
		metrics["auc"] = roc_auc_score(targets, probabilities)
	except ValueError:
		metrics["auc"] = float("nan")
	return metrics


def run_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer | None,
	criterion: nn.Module,
	device: str,
) -> Dict[str, float]:
	training = optimizer is not None
	model.train(training)

	all_targets: List[int] = []
	all_probs: List[float] = []
	all_predictions: List[int] = []
	running_loss = 0.0

	for images, targets, _paths in dataloader:
		images = images.to(device)
		targets = torch.as_tensor(targets, dtype=torch.long, device=device)

		if training:
			optimizer.zero_grad(set_to_none=True)

		logits = model(images)
		loss = criterion(logits, targets)

		if training:
			loss.backward()
			optimizer.step()

		probabilities = torch.softmax(logits, dim=1)[:, 1]
		predictions = logits.argmax(dim=1)

		running_loss += loss.item() * images.size(0)
		all_targets.extend(targets.detach().cpu().tolist())
		all_probs.extend(probabilities.detach().cpu().tolist())
		all_predictions.extend(predictions.detach().cpu().tolist())

	average_loss = running_loss / max(len(dataloader.dataset), 1)
	metrics = compute_metrics(all_targets, all_predictions, all_probs)
	metrics["loss"] = average_loss
	return metrics


def train_model(config: TrainConfig) -> Dict[str, object]:
	set_seed(config.seed)
	samples = discover_dataset(config.data_dir)
	print_dataset_summary(samples)

	train_indices, val_indices, test_indices = group_stratified_split(samples, seed=config.seed)
	train_samples = [samples[index] for index in train_indices]
	val_samples = [samples[index] for index in val_indices]
	test_samples = [samples[index] for index in test_indices]

	train_transform, eval_transform = build_transforms(config.image_size)
	train_dataset = SDNET2018Dataset(train_samples, transform=train_transform)
	val_dataset = SDNET2018Dataset(val_samples, transform=eval_transform)
	test_dataset = SDNET2018Dataset(test_samples, transform=eval_transform)

	train_loader = make_dataloader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
	val_loader = make_dataloader(val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)
	test_loader = make_dataloader(test_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers)

	model = build_model(config.model_name).to(config.device)

	class_weights = compute_class_weights(train_samples).to(config.device) if config.use_class_weights else None
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

	history: List[Dict[str, float]] = []
	best_state = None
	best_val_loss = float("inf")

	for epoch in range(1, config.epochs + 1):
		train_metrics = run_epoch(model, train_loader, optimizer, criterion, config.device)
		val_metrics = run_epoch(model, val_loader, None, criterion, config.device)
		scheduler.step(val_metrics["loss"])

		row = {f"train_{key}": value for key, value in train_metrics.items()}
		row.update({f"val_{key}": value for key, value in val_metrics.items()})
		row["epoch"] = float(epoch)
		history.append(row)

		print(
			f"Epoch {epoch:03d} | "
			f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} | "
			f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f}"
		)

		if val_metrics["loss"] < best_val_loss:
			best_val_loss = val_metrics["loss"]
			best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

	if best_state is not None:
		model.load_state_dict(best_state)

	test_metrics = run_epoch(model, test_loader, None, criterion, config.device)
	test_targets = [label for _image, label, _path in test_dataset]
	test_predictions = []
	test_probabilities = []
	model.eval()
	with torch.no_grad():
		for images, _targets, _paths in test_loader:
			images = images.to(config.device)
			logits = model(images)
			test_probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
			test_predictions.extend(logits.argmax(dim=1).cpu().tolist())
	confusion = confusion_matrix(test_targets, test_predictions).tolist()
	test_metrics["confusion_matrix"] = confusion
	print("Test metrics:", json.dumps(test_metrics, indent=2))

	return {
		"model": model,
		"history": history,
		"test_metrics": test_metrics,
		"splits": {
			"train": train_samples,
			"val": val_samples,
			"test": test_samples,
		},
	}


def grad_cam(model: nn.Module, image_tensor: torch.Tensor, target_layer: nn.Module, class_index: int | None = None) -> torch.Tensor:
	activations = None
	gradients = None

	def forward_hook(_module, _inputs, outputs):
		nonlocal activations
		activations = outputs

	def backward_hook(_module, _grad_inputs, grad_outputs):
		nonlocal gradients
		gradients = grad_outputs[0]

	forward_handle = target_layer.register_forward_hook(forward_hook)
	backward_handle = target_layer.register_full_backward_hook(backward_hook)

	try:
		model.zero_grad(set_to_none=True)
		logits = model(image_tensor)
		if class_index is None:
			class_index = int(logits.argmax(dim=1).item())
		score = logits[:, class_index].sum()
		score.backward()

		if activations is None or gradients is None:
			raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

		weights = gradients.mean(dim=(2, 3), keepdim=True)
		cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
		cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
		cam = cam.squeeze().detach()
		cam = cam - cam.min()
		cam = cam / (cam.max() + 1e-8)
		return cam
	finally:
		forward_handle.remove()
		backward_handle.remove()


def find_last_conv_layer(model: nn.Module) -> nn.Module:
	if hasattr(model, "layer4"):
		return model.layer4[-1].conv2 if hasattr(model.layer4[-1], "conv2") else model.layer4[-1]
	if isinstance(model, BaselineCNN):
		return model.features[12]
	raise ValueError("Unsupported model type for Grad-CAM")


def main() -> None:
	parser = argparse.ArgumentParser(description="SDNET2018 crack detection starter pipeline")
	parser.add_argument("--data-dir", type=Path, default=Path("./SDNET2018"))
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=15)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--model-name", type=str, default="resnet18", choices=["baseline_cnn", "resnet18", "resnet50"])
	parser.add_argument("--no-class-weights", action="store_true")
	parser.add_argument("--output-json", type=Path, default=None)
	args = parser.parse_args()

	config = TrainConfig(
		data_dir=args.data_dir,
		image_size=args.image_size,
		batch_size=args.batch_size,
		epochs=args.epochs,
		learning_rate=args.learning_rate,
		num_workers=args.num_workers,
		seed=args.seed,
		model_name=args.model_name,
		use_class_weights=not args.no_class_weights,
	)

	results = train_model(config)
	if args.output_json is not None:
		serializable = {
			"test_metrics": results["test_metrics"],
			"history": results["history"],
			"split_sizes": {key: len(value) for key, value in results["splits"].items()},
		}
		args.output_json.write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
	main()
