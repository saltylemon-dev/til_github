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
    arr = cv2.imread(path)
    if arr is None:
        raise ValueError()
    return arr


def pdf_2_png(in_pdf: Path, out_dir: Path, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(in_pdf))
    scale = dpi / 72

    image_paths: list[Path] = []
    for i in tqdm(range(len(pdf)), "| png extract |", leave=False):
        page = pdf[i]
        bitmap = page.render(scale=cast(int, scale))
        image: Image.Image = bitmap.to_pil()
        image_path = Path(out_dir / f"page_{i + 1:04d}.png")
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths


def pil_2_pdf(png_paths: list[Path], out_path, dpi: int):
    c = None
    title = out_path.stem
    for png_path in tqdm(png_paths, "| render image to pdf |", leave=False):
        img = Image.open(png_path)
        w_px, h_px = img.size
        # PNGのピクセルサイズをPDFのpointサイズへ変換
        w_pt = w_px / dpi * 72
        h_pt = h_px / dpi * 72

        if c is None:
            c = canvas.Canvas(str(out_path), pagesize=(w_pt, h_pt))
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
    pil_img = Image.fromarray(arr_rgb, mode="RGB")
    return pil_img


def pil_2_cv(img: Image.Image):
    if img.mode != "RGB":
        raise ValueError()
    arr_rgb = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return arr_bgr


def add_stripe_wrap(img_arr: NDArray):
    h, w, _ = img_arr.shape
    gray = cv2.cvtColor

    gray = cv2.cvtColor(img_arr.copy(), cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    kernel = np.ones((3, 15), dtype=np.uint8)
    graymap = cv2.dilate(gray, kernel)

    # detect right border
    col_stats = graymap.astype(np.float32).sum(axis=0)
    thresh = otsu_threshold(col_stats)
    right_border_idx = np.flatnonzero(col_stats > thresh)[-1]

    res = img_arr.copy()
    is_two_column = check_is_two_column()

    if is_two_column:
        segment_stripe(res, left=w // 2 + 10, right=right_border_idx + 10)
        segment_stripe(res, left=w - right_border_idx - 10, right=w // 2 - 10)
    else:
        raise NotImplementedError()
    return res


def check_is_two_column():
    return True


def segment_stripe(arr, top=0, bottom=None, left=0, right=None):
    arr[top:bottom, left:right] = add_stripe(arr[top:bottom, left:right])
    return arr


def add_stripe(img_arr: NDArray) -> NDArray:
    h, w, _ = img_arr.shape
    gray = cv2.cvtColor(img_arr.copy(), cv2.COLOR_BGR2GRAY)
    gray = 255 - gray

    kernel = np.ones((3, 15), dtype=np.uint8)
    arr = cv2.dilate(gray, kernel)

    row_stats = arr.astype(np.float32).sum(axis=1)
    row_stats = row_stats / 255.0 / w
    is_content_row = row_stats > 0.4

    overlay = img_arr.copy()
    in_line = False
    start = 0
    line_cnt = 0
    for i in range(h):
        if is_content_row[i] and not in_line:
            in_line = True
            start = i
            line_cnt += 1
        if not is_content_row[i] and in_line:
            in_line = False
            if line_cnt % 4 < 2:
                cv2.rectangle(overlay, (0, start - 10), (w - 1, i + 10), (0, 0, 0), -1)
    alpha = 0.4
    res = cv2.addWeighted(overlay, alpha, img_arr, 1 - alpha, 0)

    return res


def otsu_threshold(x, bins=256):
    x = np.asarray(x, dtype=float)

    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))

    # 各ビンの代表値
    values = (edges[:-1] + edges[1:]) / 2

    # 累積クラス
    w0 = np.cumsum(hist)
    w1 = x.size - w0

    # 累積平均
    sum0 = np.cumsum(hist * values)
    total = sum0[-1]

    mean0 = np.divide(sum0, w0, out=np.zeros_like(sum0, dtype=float), where=w0 > 0)

    mean1 = np.divide(total - sum0, w1, out=np.zeros_like(sum0, dtype=float), where=w1 > 0)

    # クラス間分散
    between_var = w0 * w1 * (mean0 - mean1) ** 2
    between_var[(w0 == 0) | (w1 == 0)] = 0

    return values[np.argmax(between_var)]
