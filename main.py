import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from typing import List

import fitz
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from pillow_heif import register_heif_opener
import pypdf
from pypdf import PdfReader, PdfWriter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

register_heif_opener()

app = FastAPI(
    title="Utility Tools API",
    description="Manual testing API for image, video, audio, PDF and data utilities.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

TOOL_INPUT_ERRORS = (
    ValueError,
    OSError,
    RuntimeError,
    UnidentifiedImageError,
    json.JSONDecodeError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    fitz.FileDataError,
    pypdf.errors.PyPdfError,
)


def run_tool(fn, *args):
    try:
        return fn(*args)
    except TOOL_INPUT_ERRORS:
        raise HTTPException(400, "Invalid or unsupported input file")


def empty_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


async def save_upload(file: UploadFile, tool_name: str):
    folder = UPLOAD_DIR / tool_name
    folder.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "input").name
    path = folder / f"{uuid.uuid4().hex}_{filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return path


def output_file(tool_name: str, filename: str):
    folder = RESULT_DIR / tool_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def send_file(path: Path):
    if not path.exists():
        raise HTTPException(500, f"Output was not created: {path}")
    return FileResponse(str(path), filename=path.name)


def list_results(tool_name: str):
    folder = RESULT_DIR / tool_name
    return {
        "message": "Completed successfully",
        "results": [
            str(p.relative_to(RESULT_DIR))
            for p in sorted(folder.iterdir())
            if p.is_file()
        ]
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def jpg_to_png(input_path, output_path):
    image = Image.open(input_path).convert("RGBA")
    image.save(output_path, "PNG")


def png_to_jpg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=95)


def webp_to_jpg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=95)


def jpg_to_webp(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "WEBP", quality=90)


def png_to_webp(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "WEBP", quality=90, method=6)


def bmp_to_jpg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=95)


def bmp_to_png(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "PNG")


def tiff_to_jpg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=95)


def tiff_to_png(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "PNG")


def heic_to_jpg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=95)


def heic_to_png(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "PNG")


def compress_image(input_path, output_path, quality=60):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG", quality=quality, optimize=True)


def add_watermark(input_path, output_path, text="Watermark"):
    image = Image.open(input_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    x = image.width - 150
    y = image.height - 30
    draw.text((x, y), text, fill=(255, 255, 255, 180), font=font)
    result = Image.alpha_composite(image, overlay)
    result.convert("RGB").save(output_path, "JPEG", quality=95)


def extract_images_from_pdf(pdf_path, output_dir="extracted_images"):
    os.makedirs(output_dir, exist_ok=True)
    document = fitz.open(pdf_path)
    count = 0
    for page in document:
        for image in page.get_images(full=True):
            xref = image[0]
            data = document.extract_image(xref)
            extension = data["ext"]
            count += 1
            output_path = f"{output_dir}/image_{count}.{extension}"
            with open(output_path, "wb") as file:
                file.write(data["image"])
    document.close()
    return count


def get_ffmpeg_binary():
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    venv_bin = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "ffmpeg.exe")
    if os.path.exists(venv_bin):
        return venv_bin
    return "ffmpeg"


def run_ffmpeg(command):
    cmd = list(command)
    if cmd and cmd[0] == "ffmpeg":
        cmd[0] = get_ffmpeg_binary()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg command failed with code {result.returncode}:\n{result.stderr}")


def compress_video(input_path, output_path, crf=28):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", str(crf),
        "-preset", "medium", "-c:a", "aac", "-b:a", "128k",
        output_path
    ])


def video_to_gif(input_path, output_path, fps=10, width=480):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos",
        output_path
    ])


def gif_to_mp4(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        output_path
    ])


def add_audio_to_video(video_path, audio_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path
    ])


def replace_video_audio(video_path, audio_path, output_path):
    add_audio_to_video(video_path, audio_path, output_path)


def mp4_to_mp3(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-codec:a", "libmp3lame", "-q:a", "2",
        output_path
    ])


def mp4_to_wav(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        output_path
    ])


def wav_to_mp3(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-codec:a", "libmp3lame", "-q:a", "2",
        output_path
    ])


def mp3_to_wav(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le",
        output_path
    ])


def m4a_to_mp3(input_path, output_path):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-codec:a", "libmp3lame", "-q:a", "2",
        output_path
    ])


def compress_audio(input_path, output_path, bitrate="96k"):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-codec:a", "libmp3lame", "-b:a", bitrate,
        output_path
    ])


def merge_pdfs(input_files, output_path):
    writer = PdfWriter()
    for file in input_files:
        writer.append(file)
    with open(output_path, "wb") as file:
        writer.write(file)


def split_pdf(input_path, output_dir="split_pages"):
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(input_path)
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        output_path = f"{output_dir}/page_{i}.pdf"
        with open(output_path, "wb") as file:
            writer.write(file)


def delete_pdf_pages(input_path, output_path, pages_to_delete):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    pages_to_delete = {page - 1 for page in pages_to_delete}
    for index, page in enumerate(reader.pages):
        if index not in pages_to_delete:
            writer.add_page(page)
    with open(output_path, "wb") as file:
        writer.write(file)


def pdf_to_jpg(pdf_path, output_dir="jpg_pages", dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    document = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(f"{output_dir}/page_{i + 1}.jpg")
    document.close()


def pdf_to_png(pdf_path, output_dir="png_pages", dpi=150):
    os.makedirs(output_dir, exist_ok=True)
    document = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(document):
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(f"{output_dir}/page_{i + 1}.png")
    document.close()


def jpg_to_pdf(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "PDF")


def png_to_pdf(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "PDF")


def images_to_pdf(image_files, output_path):
    images = [Image.open(file).convert("RGB") for file in image_files]
    if not images:
        raise ValueError("No images provided.")
    first = images[0]
    remaining = images[1:]
    first.save(output_path, "PDF", save_all=True, append_images=remaining)


def compress_pdf(input_path, output_path):
    document = fitz.open(input_path)
    document.save(output_path, garbage=4, deflate=True, clean=True)
    document.close()


def json_to_csv(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    dataframe = pd.json_normalize(data)
    dataframe.to_csv(output_path, index=False)


def csv_to_json(input_path, output_path):
    dataframe = pd.read_csv(input_path)
    if dataframe.empty:
        raise ValueError("Empty or invalid CSV file")
    dataframe.to_json(output_path, orient="records", indent=4)


def csv_to_excel(input_path, output_path):
    dataframe = pd.read_csv(input_path)
    if dataframe.empty:
        raise ValueError("Empty or invalid CSV file")
    dataframe.to_excel(output_path, index=False)


def excel_to_csv(input_path, output_path, sheet_name=0):
    dataframe = pd.read_excel(input_path, sheet_name=sheet_name)
    dataframe.to_csv(output_path, index=False)


# ============================================================
# API ENDPOINTS
# ============================================================

# ---------------- IMAGE ----------------

@app.post("/jpg-to-png")
async def jpg_to_png_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "jpg_to_png")
    out = output_file("jpg_to_png", "output.png")
    run_tool(jpg_to_png, str(src), str(out))
    return send_file(out)


@app.post("/png-to-jpg")
async def png_to_jpg_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "png_to_jpg")
    out = output_file("png_to_jpg", "output.jpg")
    run_tool(png_to_jpg, str(src), str(out))
    return send_file(out)


@app.post("/webp-to-jpg")
async def webp_to_jpg_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "webp_to_jpg")
    out = output_file("webp_to_jpg", "output.jpg")
    run_tool(webp_to_jpg, str(src), str(out))
    return send_file(out)


@app.post("/jpg-to-webp")
async def jpg_to_webp_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "jpg_to_webp")
    out = output_file("jpg_to_webp", "output.webp")
    run_tool(jpg_to_webp, str(src), str(out))
    return send_file(out)


@app.post("/png-to-webp")
async def png_to_webp_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "png_to_webp")
    out = output_file("png_to_webp", "output.webp")
    run_tool(png_to_webp, str(src), str(out))
    return send_file(out)


@app.post("/bmp-to-jpg")
async def bmp_to_jpg_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "bmp_to_jpg")
    out = output_file("bmp_to_jpg", "output.jpg")
    run_tool(bmp_to_jpg, str(src), str(out))
    return send_file(out)


@app.post("/bmp-to-png")
async def bmp_to_png_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "bmp_to_png")
    out = output_file("bmp_to_png", "output.png")
    run_tool(bmp_to_png, str(src), str(out))
    return send_file(out)


@app.post("/tiff-to-jpg")
async def tiff_to_jpg_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "tiff_to_jpg")
    out = output_file("tiff_to_jpg", "output.jpg")
    run_tool(tiff_to_jpg, str(src), str(out))
    return send_file(out)


@app.post("/tiff-to-png")
async def tiff_to_png_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "tiff_to_png")
    out = output_file("tiff_to_png", "output.png")
    run_tool(tiff_to_png, str(src), str(out))
    return send_file(out)


@app.post("/heic-to-jpg")
async def heic_to_jpg_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "heic_to_jpg")
    out = output_file("heic_to_jpg", "output.jpg")
    run_tool(heic_to_jpg, str(src), str(out))
    return send_file(out)


@app.post("/heic-to-png")
async def heic_to_png_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "heic_to_png")
    out = output_file("heic_to_png", "output.png")
    run_tool(heic_to_png, str(src), str(out))
    return send_file(out)


@app.post("/compress-image")
async def compress_image_endpoint(
    file: UploadFile = File(...),
    quality: int = Form(60)
):
    if not 1 <= quality <= 100:
        raise HTTPException(400, "quality must be between 1 and 100")

    src = await save_upload(file, "compress_image")
    out = output_file("compress_image", "compressed.jpg")
    run_tool(compress_image, str(src), str(out), quality)
    return send_file(out)


@app.post("/add-watermark")
async def add_watermark_endpoint(
    file: UploadFile = File(...),
    text: str = Form("Watermark")
):
    src = await save_upload(file, "add_watermark")
    out = output_file("add_watermark", "watermarked.jpg")
    run_tool(add_watermark, str(src), str(out), text)
    return send_file(out)


@app.post("/extract-images-from-pdf")
async def extract_images_from_pdf_endpoint(file: UploadFile = File(...)):
    folder = RESULT_DIR / "extract_images_from_pdf"
    empty_folder(folder)

    src = await save_upload(file, "extract_images_from_pdf")
    count = run_tool(extract_images_from_pdf, str(src), str(folder))

    return {
        "message": "Images extracted successfully",
        "count": count,
        "results": [
            str(p.relative_to(RESULT_DIR))
            for p in sorted(folder.iterdir())
            if p.is_file()
        ]
    }


# ---------------- VIDEO ----------------

@app.post("/compress-video")
async def compress_video_endpoint(
    file: UploadFile = File(...),
    crf: int = Form(28)
):
    if not 0 <= crf <= 51:
        raise HTTPException(400, "crf must be between 0 and 51")

    src = await save_upload(file, "compress_video")
    out = output_file("compress_video", "compressed.mp4")
    run_tool(compress_video, str(src), str(out), crf)
    return send_file(out)


@app.post("/video-to-gif")
async def video_to_gif_endpoint(
    file: UploadFile = File(...),
    fps: int = Form(10),
    width: int = Form(480)
):
    if fps <= 0:
        raise HTTPException(400, "fps must be a positive integer")
    if width <= 0:
        raise HTTPException(400, "width must be a positive integer")

    src = await save_upload(file, "video_to_gif")
    out = output_file("video_to_gif", "output.gif")
    run_tool(video_to_gif, str(src), str(out), fps, width)
    return send_file(out)


@app.post("/gif-to-mp4")
async def gif_to_mp4_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "gif_to_mp4")
    out = output_file("gif_to_mp4", "output.mp4")
    run_tool(gif_to_mp4, str(src), str(out))
    return send_file(out)


@app.post("/add-audio-to-video")
async def add_audio_to_video_endpoint(
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    video_src = await save_upload(video, "add_audio_to_video")
    audio_src = await save_upload(audio, "add_audio_to_video")
    out = output_file("add_audio_to_video", "output.mp4")

    run_tool(
        add_audio_to_video,
        str(video_src),
        str(audio_src),
        str(out)
    )

    return send_file(out)


@app.post("/replace-video-audio")
async def replace_video_audio_endpoint(
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    video_src = await save_upload(video, "replace_video_audio")
    audio_src = await save_upload(audio, "replace_video_audio")
    out = output_file("replace_video_audio", "output.mp4")

    run_tool(
        replace_video_audio,
        str(video_src),
        str(audio_src),
        str(out)
    )

    return send_file(out)


# ---------------- AUDIO ----------------

@app.post("/mp4-to-mp3")
async def mp4_to_mp3_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "mp4_to_mp3")
    out = output_file("mp4_to_mp3", "output.mp3")
    run_tool(mp4_to_mp3, str(src), str(out))
    return send_file(out)


@app.post("/mp4-to-wav")
async def mp4_to_wav_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "mp4_to_wav")
    out = output_file("mp4_to_wav", "output.wav")
    run_tool(mp4_to_wav, str(src), str(out))
    return send_file(out)


@app.post("/wav-to-mp3")
async def wav_to_mp3_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "wav_to_mp3")
    out = output_file("wav_to_mp3", "output.mp3")
    run_tool(wav_to_mp3, str(src), str(out))
    return send_file(out)


@app.post("/mp3-to-wav")
async def mp3_to_wav_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "mp3_to_wav")
    out = output_file("mp3_to_wav", "output.wav")
    run_tool(mp3_to_wav, str(src), str(out))
    return send_file(out)


@app.post("/m4a-to-mp3")
async def m4a_to_mp3_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "m4a_to_mp3")
    out = output_file("m4a_to_mp3", "output.mp3")
    run_tool(m4a_to_mp3, str(src), str(out))
    return send_file(out)


@app.post("/compress-audio")
async def compress_audio_endpoint(
    file: UploadFile = File(...),
    bitrate: str = Form("96k")
):
    bitrate_val = bitrate.strip()
    if bitrate_val.isdigit():
        bitrate_val = f"{bitrate_val}k"
    if not re.match(r"^\d+k$", bitrate_val):
        raise HTTPException(400, "Invalid bitrate format")

    src = await save_upload(file, "compress_audio")
    out = output_file("compress_audio", "compressed.mp3")
    run_tool(compress_audio, str(src), str(out), bitrate_val)
    return send_file(out)


# ---------------- PDF ----------------

@app.post("/merge-pdfs")
async def merge_pdfs_endpoint(files: List[UploadFile] = File(...)):
    if len(files) < 2:
        raise HTTPException(400, "Upload at least 2 PDF files")

    paths = []
    for file in files:
        path = await save_upload(file, "merge_pdfs")
        paths.append(str(path))

    out = output_file("merge_pdfs", "merged.pdf")
    run_tool(merge_pdfs, paths, str(out))

    return send_file(out)


@app.post("/split-pdf")
async def split_pdf_endpoint(file: UploadFile = File(...)):
    folder = RESULT_DIR / "split_pdf"
    empty_folder(folder)

    src = await save_upload(file, "split_pdf")
    run_tool(split_pdf, str(src), str(folder))

    return list_results("split_pdf")


@app.post("/delete-pdf-pages")
async def delete_pdf_pages_endpoint(
    file: UploadFile = File(...),
    pages: str = Form(...)
):
    try:
        page_numbers = [
            int(x.strip())
            for x in pages.split(",")
            if x.strip()
        ]
    except ValueError:
        raise HTTPException(400, "Use page numbers like: 2,4,6")

    if not page_numbers:
        raise HTTPException(400, "Use page numbers like: 2,4,6")

    src = await save_upload(file, "delete_pdf_pages")

    try:
        reader = pypdf.PdfReader(str(src))
        total_pages = len(reader.pages)
    except Exception:
        raise HTTPException(400, "Invalid or unsupported input file")

    if any(p < 1 or p > total_pages for p in page_numbers):
        raise HTTPException(400, "Invalid page numbers")

    if set(page_numbers).issuperset(range(1, total_pages + 1)):
        raise HTTPException(400, "Cannot delete all pages")

    out = output_file("delete_pdf_pages", "output.pdf")
    run_tool(delete_pdf_pages, str(src), str(out), page_numbers)

    return send_file(out)


@app.post("/pdf-to-jpg")
async def pdf_to_jpg_endpoint(
    file: UploadFile = File(...),
    dpi: int = Form(150)
):
    if not 1 <= dpi <= 600:
        raise HTTPException(400, "dpi must be between 1 and 600")

    folder = RESULT_DIR / "pdf_to_jpg"
    empty_folder(folder)

    src = await save_upload(file, "pdf_to_jpg")
    run_tool(pdf_to_jpg, str(src), str(folder), dpi)

    return list_results("pdf_to_jpg")


@app.post("/pdf-to-png")
async def pdf_to_png_endpoint(
    file: UploadFile = File(...),
    dpi: int = Form(150)
):
    if not 1 <= dpi <= 600:
        raise HTTPException(400, "dpi must be between 1 and 600")

    folder = RESULT_DIR / "pdf_to_png"
    empty_folder(folder)

    src = await save_upload(file, "pdf_to_png")
    run_tool(pdf_to_png, str(src), str(folder), dpi)

    return list_results("pdf_to_png")


@app.post("/jpg-to-pdf")
async def jpg_to_pdf_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "jpg_to_pdf")
    out = output_file("jpg_to_pdf", "output.pdf")
    run_tool(jpg_to_pdf, str(src), str(out))
    return send_file(out)


@app.post("/png-to-pdf")
async def png_to_pdf_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "png_to_pdf")
    out = output_file("png_to_pdf", "output.pdf")
    run_tool(png_to_pdf, str(src), str(out))
    return send_file(out)


@app.post("/images-to-pdf")
async def images_to_pdf_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "Upload at least one image")

    paths = []
    for file in files:
        path = await save_upload(file, "images_to_pdf")
        paths.append(str(path))

    out = output_file("images_to_pdf", "images.pdf")
    run_tool(images_to_pdf, paths, str(out))

    return send_file(out)


@app.post("/compress-pdf")
async def compress_pdf_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "compress_pdf")
    out = output_file("compress_pdf", "compressed.pdf")
    run_tool(compress_pdf, str(src), str(out))
    return send_file(out)


# ---------------- DATA ----------------

@app.post("/json-to-csv")
async def json_to_csv_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "json_to_csv")
    out = output_file("json_to_csv", "output.csv")
    run_tool(json_to_csv, str(src), str(out))
    return send_file(out)


@app.post("/csv-to-json")
async def csv_to_json_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "csv_to_json")
    out = output_file("csv_to_json", "output.json")
    run_tool(csv_to_json, str(src), str(out))
    return send_file(out)


@app.post("/csv-to-excel")
async def csv_to_excel_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "csv_to_excel")
    out = output_file("csv_to_excel", "output.xlsx")
    run_tool(csv_to_excel, str(src), str(out))
    return send_file(out)


@app.post("/excel-to-csv")
async def excel_to_csv_endpoint(file: UploadFile = File(...)):
    src = await save_upload(file, "excel_to_csv")
    out = output_file("excel_to_csv", "output.csv")
    run_tool(excel_to_csv, str(src), str(out))
    return send_file(out)


# ---------------- HOME ----------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Utility Tools API is running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
