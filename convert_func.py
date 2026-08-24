from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tqdm import tqdm


def load_cv(path: Path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError()
    return img


def pdf_2_png(in_pdf: Path, out_dir: Path, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(in_pdf))
    scale = dpi / 72

    for i in tqdm(range(len(pdf)), "| png extract |", leave=False):
        page = pdf[i]
        bitmap = page.render(scale=cast(int, scale))
        image: Image.Image = bitmap.to_pil()
        image.save(out_dir / f"page_{i + 1:04d}.png")
    return len(pdf)


def pil_2_pdf(images: list[Image.Image], path, dpi: int):
    c = None
    title = path.stem
    for img in images:
        w_px, h_px = img.size
        # PNGのピクセルサイズをPDFのpointサイズへ変換
        w_pt = w_px / dpi * 72
        h_pt = h_px / dpi * 72

        if c is None:
            c = canvas.Canvas(str(path), pagesize=(w_pt, h_pt))
            c.setTitle(title)
        else:
            c.setPageSize((w_pt, h_pt))

        c.drawImage(ImageReader(img), 0, 0, width=w_pt, height=h_pt)
        c.showPage()

    if c is not None:
        c.save()


def cv_2_pil(arr: NDArray) -> Image.Image:
    if arr.ndim != 3:
        raise NotImplementedError()
    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    res = Image.fromarray(arr_rgb, mode="RGB")
    return res


def add_stripe(img: NDArray) -> NDArray:
    h, w, _ = img.shape
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = 255 - gray

    height = 11
    kernel = np.zeros((height, height), np.uint8)
    kernel[height // 2, :] = 1
    arr = cv2.dilate(gray, kernel)

    row_stats = arr.astype(np.float32).sum(axis=1)
    row_stats = row_stats / 255.0 / w
    is_letter = row_stats > 0.4

    overlay = img.copy()
    in_line = False
    start = 0
    line_cnt = 0
    for i in range(h):
        if is_letter[i] and in_line:
            pass
        if is_letter[i] and not in_line:
            in_line = True
            start = i
            line_cnt += 1
        if not is_letter[i] and not in_line:
            pass
        if not is_letter[i] and in_line:
            in_line = False
            if line_cnt % 4 < 2:
                cv2.rectangle(overlay, (3 * w // 20, start - 10), (17 * w // 20, i + 10), (0, 0, 0), -1)
    alpha = 0.4
    res = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    return res
