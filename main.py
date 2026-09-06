from pathlib import Path

import cv2
from tqdm import tqdm

from convert_func import add_stripe_wrap, load_cv, pdf_2_png, pil_2_pdf


def main():
    dpi = 300
    input_pdf_root = Path("./inputs")
    output_pdf_root = Path("./outputs")
    diff_path = get_diff_rel_paths(input_pdf_root, output_pdf_root)
    for rel_path in tqdm(diff_path, "| add stripe on pdf |"):
        input_path = input_pdf_root / rel_path
        output_path = output_pdf_root / rel_path
        stripe_on_pdf(input_path, output_path, dpi=dpi)


def get_diff_rel_paths(srcdir: Path, dstdir: Path):
    if not srcdir.is_dir() or not dstdir.is_dir():
        raise ValueError()
    src_rel_paths = [path.relative_to(srcdir) for path in srcdir.rglob("*.pdf")]
    dst_rel_paths = [path.relative_to(dstdir) for path in dstdir.rglob("*.pdf")]
    diff = list(set(src_rel_paths) - set(dst_rel_paths))
    return diff


def stripe_on_pdf(in_pdf: Path, out_pdf: Path, dpi: int):
    buffer_dir = Path("./__buffer")
    png_paths = pdf_2_png(in_pdf, buffer_dir, dpi)
    for png_path in tqdm(png_paths, "| add stripe on png |", leave=False):
        arr = load_cv(png_path)
        arr = add_stripe_wrap(arr.copy())
        cv2.imwrite(png_path, arr)

    (out_pdf.parent).mkdir(parents=True, exist_ok=True)
    pil_2_pdf(png_paths, out_pdf, dpi)


if __name__ == "__main__":
    main()
