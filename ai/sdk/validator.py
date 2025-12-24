import torch
from sdk.contracts import DatasetValidatorAdapter


class DatasetValidationError(Exception):
    pass


class DatasetValidator(DatasetValidatorAdapter):
    def __init__(self, dataset, max_samples: int | None = None):
        self.dataset = dataset
        self.max_samples = max_samples or len(dataset)

    def validate(self):
        self._validate_dataset_level()

        for idx in range(min(len(self.dataset), self.max_samples)):
            self._validate_sample(idx)

    def _validate_dataset_level(self):
        if len(self.dataset) == 0:
            raise DatasetValidationError("Датасет пуст")

        if not hasattr(self.dataset, "num_classes") or self.dataset.num_classes < 2:
            raise DatasetValidationError("num_classes некорректен")

        if not hasattr(self.dataset, "num_attr_classes") or self.dataset.num_attr_classes < 1:
            raise DatasetValidationError("num_attr_classes некорректен")

        if not hasattr(self.dataset, "class_names"):
            raise DatasetValidationError("class_names отсутствуют")

    def _validate_sample(self, idx: int):
        try:
            image, target, _ = self.dataset[idx]
        except Exception as e:
            raise DatasetValidationError(f"[{idx}] Ошибка загрузки sample: {e}")

        self._validate_image(image, idx)
        self._validate_target(target, image.shape, idx)

    def _validate_image(self, image, idx):
        if not isinstance(image, torch.Tensor):
            raise DatasetValidationError(f"[{idx}] image не Tensor")

        if image.ndim != 3:
            raise DatasetValidationError(f"[{idx}] image.ndim != 3")

        if image.shape[1] == 0 or image.shape[2] == 0:
            raise DatasetValidationError(f"[{idx}] image пустое")

        if torch.isnan(image).any():
            raise DatasetValidationError(f"[{idx}] NaN в image")

    def _validate_target(self, target: dict, image_shape, idx: int):
        required = {"boxes", "labels", "masks", "attr_labels"}
        missing = required - target.keys()
        if missing:
            raise DatasetValidationError(f"[{idx}] target missing {missing}")

        boxes = target["boxes"]
        labels = target["labels"]
        masks = target["masks"]
        attrs = target["attr_labels"]

        n = len(boxes)
        if not (len(labels) == len(masks) == len(attrs) == n):
            raise DatasetValidationError(f"[{idx}] Несовпадение размеров target")

        for i in range(n):
            self._validate_box(
                boxes[i],
                image_shape=image_shape,
                idx=idx,
                obj=i,
            )
            self._validate_label(labels[i], idx, i)
            self._validate_attr(attrs[i], idx, i)
            self._validate_mask(masks[i], idx, i)

    def _validate_box(self, box, image_shape, idx, obj):
        x1, y1, x2, y2 = box.tolist()
        h, w = image_shape[1:]

        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise DatasetValidationError(f"[{idx}:{obj}] bbox вне изображения: {box.tolist()}")

    def _validate_label(self, label, idx, obj):
        if not (1 <= int(label) < self.dataset.num_classes):
            raise DatasetValidationError(f"[{idx}:{obj}] label={label} вне диапазона")

    def _validate_attr(self, attr, idx, obj):
        if not (0 <= int(attr) < self.dataset.num_attr_classes):
            raise DatasetValidationError(f"[{idx}:{obj}] attr_label={attr} вне диапазона")

    def _validate_mask(self, mask, idx, obj):
        if mask.sum() == 0:
            raise DatasetValidationError(f"[{idx}:{obj}] пустая mask")
