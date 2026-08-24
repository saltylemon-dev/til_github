from pathlib import Path

from PIL import Image
from tqdm import tqdm

from convert_func import add_stripe, cv_2_pil, load_cv, pdf_2_png, pil_2_pdf


def main():
    dpi = 300
    pdf_dir = Path("./inputs")
    outdir = Path("./outputs/striped")

    paths = list(pdf_dir.glob("*.pdf"))
    for path in tqdm(paths, "| add atripe on pdf |"):
        stripe_on_pdf(path, outdir, dpi=dpi)


def stripe_on_pdf(in_pdf: Path, outdir: Path, dpi: int):
    png_out = Path("./outputs/__tmp")
    num = pdf_2_png(in_pdf, png_out, dpi)
    images: list[Image.Image] = []

    for i in range(num):
        path = png_out / f"page_{i + 1:04d}.png"
        arr = load_cv(path)
        arr = add_stripe(arr.copy())
        img = cv_2_pil(arr)
        images.append(img.copy())

    outdir.mkdir(parents=True, exist_ok=True)
    pil_2_pdf(images, outdir / f"{in_pdf.name}", dpi)
    print("-- stripe added --")


if __name__ == "__main__":
    main()
