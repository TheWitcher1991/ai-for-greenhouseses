import numpy as np
import torch


class ComposeTransforms:
    def __call__(self, image, target):
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            boxes = target["boxes"]
            boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
            target["boxes"] = boxes

            masks = target["masks"]
            if isinstance(masks, torch.Tensor):
                masks = torch.flip(masks, dims=[2])  # axis=2 для ширины
            else:
                masks = np.flip(masks, axis=2).copy()
            target["masks"] = masks
        return image, target
