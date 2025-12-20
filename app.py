import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from infer import predict


class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Segmentation")
        self.root.geometry("1000x700")

        self.image_path = None
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.load_btn = tk.Button(btn_frame, text="Загрузить изображение", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.predict_btn = tk.Button(btn_frame, text="Сегментировать", command=self.run_prediction)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if path:
            self.image_path = path
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((900, 600))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)

    def run_prediction(self):
        if not self.image_path:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        results = predict(self.image_path, score_threshold=0.5)

        if not results:
            messagebox.showinfo("Результат", "Объекты не найдены.")
            return

        img = cv2.imread(self.image_path)
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
            cv2.putText(
                overlay, f"{label} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        overlay = Image.fromarray(overlay)
        overlay.thumbnail((900, 600))
        self.tk_img = ImageTk.PhotoImage(overlay)
        self.img_label.config(image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
