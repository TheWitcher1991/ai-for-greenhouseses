import numpy as np
import torch

from .contracts import TransformAdapter


class ComposeTransforms(TransformAdapter):
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


# class ComposeTransforms(TransformAdapter):
#     def __init__(self, target_size=(512, 512)):
#         self.target_size = target_size

#     def __call__(self, image, target):
#         orig_h, orig_w = image.shape[:2]
#         new_h, new_w = self.target_size
#         image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#         boxes = (
#             target["boxes"].clone()
#             if isinstance(target["boxes"], torch.Tensor)
#             else np.array(target["boxes"], dtype=np.float32)
#         )
#         scale_x = new_w / orig_w
#         scale_y = new_h / orig_h
#         boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
#         boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
#         if isinstance(boxes, torch.Tensor):
#             target["boxes"] = boxes.detach().clone()
#         else:
#             target["boxes"] = torch.tensor(boxes, dtype=torch.float32)

#         masks = target["masks"]
#         if isinstance(masks, torch.Tensor):
#             masks_resized = []
#             for m in masks:
#                 m_np = m.cpu().numpy()
#                 m_resized = cv2.resize(m_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#                 masks_resized.append(torch.tensor(m_resized, dtype=torch.float32))
#             masks = torch.stack(masks_resized, dim=0)
#         else:
#             masks_resized = []
#             for m in masks:
#                 masks_resized.append(cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
#             masks = np.stack(masks_resized, axis=0)
#         target["masks"] = masks

#         if np.random.rand() > 0.5:
#             image_resized = np.fliplr(image_resized).copy()
#             boxes = target["boxes"]
#             boxes[:, [0, 2]] = new_w - boxes[:, [2, 0]]
#             target["boxes"] = boxes

#             masks = target["masks"]
#             if isinstance(masks, torch.Tensor):
#                 masks = torch.flip(masks, dims=[2])
#             else:
#                 masks = np.flip(masks, axis=2).copy()
#             target["masks"] = masks

#         return image_resized, target
