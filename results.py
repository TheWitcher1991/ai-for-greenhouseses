import cv2
import matplotlib.pyplot as plt
import numpy as np

from infer import predict

image_path = "data/test.jpg"

results = predict(image_path, score_threshold=0.5)

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

overlay = img.copy()

for r in results:
    mask = r["mask"]
    label = r["label"]
    score = r["score"]

    color = np.random.randint(0, 255, size=3)
    mask_rgb = np.zeros_like(img)
    mask_rgb[:, :, 0] = mask * color[0]
    mask_rgb[:, :, 1] = mask * color[1]
    mask_rgb[:, :, 2] = mask * color[2]

    overlay = cv2.addWeighted(overlay, 1.0, mask_rgb, 0.5, 0)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        continue
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)

    cv2.putText(overlay, f"{label} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

plt.figure(figsize=(12, 8))
plt.imshow(overlay)
plt.axis("off")
plt.show()
