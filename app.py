"""
ImageEnhance - Digital Image Restoration and Quality Enhancement
Run with:  streamlit run app.py

Sections of this file
  1. Settings and folders
  2. Brand assets (logo, favicon) and icons
  3. Image processing functions
  4. Quality metrics and file saving
  5. Page styling
  6. Page layouts (Workspace, Saved Images, Experiment History, Help)
  7. main()
"""

import base64
import csv
import hashlib
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =====================================================================
# 1. SETTINGS AND FOLDERS
# =====================================================================

APP_NAME = "ImageEnhance"
TAGLINE = "Restore Clarity. Reveal Details."
PROJECT_TITLE = "Digital Image Restoration and Quality Enhancement"

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUTS_DIR = BASE_DIR / "outputs"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
ASSETS_DIR = BASE_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "logo.svg"
MARK_PATH = ASSETS_DIR / "logo_mark.svg"
FAVICON_PATH = ASSETS_DIR / "favicon.png"
LOG_FILE = EXPERIMENTS_DIR / "experiment_log.csv"

ALLOWED_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
ALLOWED_SUFFIXES = {"." + t for t in ALLOWED_TYPES}
MAX_SIDE = 1280  # larger images are shrunk for processing (the saved original is untouched)

LOG_FIELDS = [
    "timestamp", "experiment", "source_image", "degradation", "restoration",
    "input_size", "output_size", "psnr_db", "ssim", "output_file", "observations",
]


def prepare_folders():
    """Create the project folders. Never deletes anything.
    Returns a list of paths that exist as FILES where a folder is needed."""
    problems = []
    for folder in (IMAGES_DIR, OUTPUTS_DIR, EXPERIMENTS_DIR, ASSETS_DIR):
        if folder.exists() and not folder.is_dir():
            problems.append(folder)
        else:
            folder.mkdir(parents=True, exist_ok=True)
    return problems


# =====================================================================
# 2. BRAND ASSETS AND ICONS
# =====================================================================

# Logo mark: a picture frame; blocky pixels on the left become a smooth
# landscape on the right; a sparkle marks the enhancement.
MARK_INNER = """<defs><linearGradient id="ieg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#2563EB"/><stop offset="1" stop-color="#06B6D4"/></linearGradient></defs>
<rect width="64" height="64" rx="14" fill="url(#ieg)"/>
<rect x="11" y="14" width="42" height="36" rx="4" fill="none" stroke="#FFFFFF" stroke-width="3"/>
<g fill="#FFFFFF">
<rect x="15" y="43" width="5" height="5" opacity=".95"/><rect x="21" y="43" width="5" height="5" opacity=".95"/>
<rect x="27" y="43" width="5" height="5" opacity=".95"/><rect x="21" y="37" width="5" height="5" opacity=".7"/>
<rect x="27" y="37" width="5" height="5" opacity=".7"/><rect x="27" y="31" width="5" height="5" opacity=".45"/>
<path d="M34 48 C37 38 40 34 43 40 C45 43 47 38 49 36 L49 48 Z" opacity=".95"/>
<path d="M43 19 L44.4 22.6 L48 24 L44.4 25.4 L43 29 L41.6 25.4 L38 24 L41.6 22.6 Z"/>
</g>"""

MARK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" '
    'role="img" aria-label="ImageEnhance logo mark">' + MARK_INNER + "</svg>"
)

LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 270 64" width="270" height="64" '
    'role="img" aria-label="ImageEnhance logo">' + MARK_INNER +
    '<text x="78" y="41" font-family="Inter, \'Segoe UI\', Arial, sans-serif" font-size="30" '
    'font-weight="700" fill="#14213D">Image<tspan fill="#2563EB">Enhance</tspan></text></svg>'
)


def _bezier(p0, p1, p2, p3, steps=12):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0]
             + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
        y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1]
             + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])
        pts.append((x, y))
    return pts


def make_favicon(path):
    """Draw the logo mark as a 256x256 PNG with Pillow (same shapes as the SVG)."""
    size, k = 256, 4  # k = scale from the 64-unit SVG grid
    yy, xx = np.mgrid[0:size, 0:size]
    t = (xx + yy) / (2.0 * (size - 1))
    blue, cyan = np.array([37, 99, 235]), np.array([6, 182, 212])
    rgb = (blue[None, None, :] * (1 - t[..., None]) + cyan[None, None, :] * t[..., None])
    base = Image.fromarray(rgb.astype(np.uint8), "RGB").convert("RGBA")

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=14 * k, fill=255)
    base.putalpha(mask)

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle([11 * k, 14 * k, 53 * k, 50 * k], radius=4 * k,
                        outline=(255, 255, 255, 255), width=3 * k)
    blocks = [(15, 43, .95), (21, 43, .95), (27, 43, .95), (21, 37, .7), (27, 37, .7), (27, 31, .45)]
    for x, y, op in blocks:
        d.rectangle([x * k, y * k, (x + 5) * k, (y + 5) * k], fill=(255, 255, 255, int(255 * op)))
    mountain = [(34, 48)]
    mountain += _bezier((34, 48), (37, 38), (40, 34), (43, 40))
    mountain += _bezier((43, 40), (45, 43), (47, 38), (49, 36))
    mountain += [(49, 48)]
    d.polygon([(x * k, y * k) for x, y in mountain], fill=(255, 255, 255, 242))
    star = [(43, 19), (44.4, 22.6), (48, 24), (44.4, 25.4), (43, 29), (41.6, 25.4), (38, 24), (41.6, 22.6)]
    d.polygon([(x * k, y * k) for x, y in star], fill=(255, 255, 255, 255))

    Image.alpha_composite(base, overlay).save(path, format="PNG")


def ensure_assets():
    """Create logo and favicon files only if they do not exist yet."""
    if not ASSETS_DIR.is_dir():
        return
    for path, text in ((LOGO_PATH, LOGO_SVG), (MARK_PATH, MARK_SVG)):
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    if not FAVICON_PATH.exists():
        try:
            make_favicon(FAVICON_PATH)
        except Exception:
            pass


def file_data_uri(path, mime, fallback_text=None):
    """Return a data: URI for a local file (used to show SVG logos reliably)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        if fallback_text is None:
            return ""
        data = fallback_text.encode("utf-8")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# Line icons in the style of the Lucide set (ISC licence).
ICON_PATHS = {
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "restore": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "noise": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "sharpen": '<circle cx="12" cy="12" r="3"/><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
    "contrast": '<circle cx="12" cy="12" r="10"/><path d="M12 18a6 6 0 0 0 0-12v12z"/>',
    "upscale": '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "compare": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 3v18"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "flask": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>',
    "sliders": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    "sparkle": '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
}


def icon_html(name, color="#2563EB", size=22):
    """Small inline SVG icon, embedded as an image so Streamlit displays it reliably."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICON_PATHS[name]}</svg>'
    )
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="{uri}" width="{size}" height="{size}" alt="" class="ie-icon">'


# =====================================================================
# 3. IMAGE PROCESSING
# =====================================================================

DEFAULTS = {
    "degradation": "None",
    "noise_level": 20,
    "sp_amount": 0.04,
    "blur_strength": 7,
    "down_factor": 2,
    "filter_method": "None",
    "median_k": 5,
    "gauss_k": 5,
    "bilateral_sigma": 75,
    "sharpening": False,
    "sharp_strength": 0.7,
    "sharp_radius": 2.0,
    "contrast": False,
    "clahe_clip": 2.0,
    "upscale_factor": "Off",
}

DEGRADATIONS = ["None", "Gaussian Noise", "Salt-and-Pepper Noise", "Gaussian Blur", "Low Resolution"]
FILTERS = ["None", "Median Filter", "Gaussian Filter", "Bilateral Filter"]
UPSCALES = ["Off", "2x", "3x", "4x"]

PRESETS = {
    "Experiment 1: Gaussian noise removal": {
        "degradation": "Gaussian Noise", "noise_level": 25, "filter_method": "Gaussian Filter",
    },
    "Experiment 2: Salt-and-pepper noise removal": {
        "degradation": "Salt-and-Pepper Noise", "sp_amount": 0.05,
        "filter_method": "Median Filter", "median_k": 5,
    },
    "Experiment 3: Gaussian blur reduction": {
        "degradation": "Gaussian Blur", "blur_strength": 7,
        "sharpening": True, "sharp_strength": 1.2, "sharp_radius": 2.0,
    },
    "Experiment 4: Low-resolution upscaling": {
        "degradation": "Low Resolution", "down_factor": 2, "upscale_factor": "2x",
    },
}
NO_PRESET = "Custom (no preset)"


@st.cache_data(show_spinner=False)
def decode_image(data):
    """Decode image bytes to an RGB array. Returns (array, full_size, was_resized)."""
    img = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    full_size = img.size
    resized = False
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        resized = True
    return np.array(img), full_size, resized


def degrade_image(original, s):
    """Apply the chosen degradation. Returns (degraded, reference).
    `reference` is the clean image that matches the degraded scene."""
    mode = s["degradation"]
    rng = np.random.default_rng(42)  # fixed seed: same settings -> same degraded image
    reference = original

    if mode == "Gaussian Noise":
        noise = rng.normal(0, s["noise_level"], original.shape)
        degraded = np.clip(original.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif mode == "Salt-and-Pepper Noise":
        degraded = original.copy()
        values = rng.random(original.shape[:2])
        degraded[values < s["sp_amount"] / 2] = 0
        degraded[values > 1 - s["sp_amount"] / 2] = 255
    elif mode == "Gaussian Blur":
        k = int(s["blur_strength"])
        degraded = cv2.GaussianBlur(original, (k, k), 0)
    elif mode == "Low Resolution":
        f = int(s["down_factor"])
        h, w = original.shape[:2]
        if min(h, w) < f * 8:
            raise ValueError("This image is too small for the chosen downscaling factor.")
        h2, w2 = (h // f) * f, (w // f) * f
        reference = original[:h2, :w2]  # crop so the size divides evenly
        degraded = cv2.resize(reference, (w2 // f, h2 // f), interpolation=cv2.INTER_AREA)
    else:
        degraded = original.copy()
    return degraded, reference


def apply_noise_filter(img, method, s):
    if method == "Median Filter":
        return cv2.medianBlur(img, int(s["median_k"]))
    if method == "Gaussian Filter":
        k = int(s["gauss_k"])
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "Bilateral Filter":
        return cv2.bilateralFilter(img, 9, float(s["bilateral_sigma"]), float(s["bilateral_sigma"]))
    return img.copy()


def unsharp_mask(img, amount, radius):
    blurred = cv2.GaussianBlur(img, (0, 0), float(radius))
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def apply_clahe(img, clip):
    """CLAHE on the lightness channel only, so colours are preserved."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(light), a, b)), cv2.COLOR_LAB2RGB)


def upscale_factor_value(s):
    return 1 if s["upscale_factor"] == "Off" else int(s["upscale_factor"][0])


def restore_image(degraded, s):
    """Run the restoration pipeline on the degraded input:
    noise filter -> sharpening -> CLAHE -> bicubic upscaling."""
    result = apply_noise_filter(degraded, s["filter_method"], s)
    if s["sharpening"]:
        result = unsharp_mask(result, s["sharp_strength"], s["sharp_radius"])
    if s["contrast"]:
        result = apply_clahe(result, s["clahe_clip"])
    f = upscale_factor_value(s)
    if f > 1:
        h, w = result.shape[:2]
        result = cv2.resize(result, (w * f, h * f), interpolation=cv2.INTER_CUBIC)
    return result


def degradation_text(s):
    mode = s["degradation"]
    if mode == "Gaussian Noise":
        return f"Gaussian noise (strength {s['noise_level']})"
    if mode == "Salt-and-Pepper Noise":
        return f"Salt-and-pepper noise ({s['sp_amount'] * 100:.0f}% of pixels)"
    if mode == "Gaussian Blur":
        return f"Gaussian blur ({s['blur_strength']}x{s['blur_strength']} kernel)"
    if mode == "Low Resolution":
        return f"Low resolution (downscaled by {s['down_factor']})"
    return "None"


def restoration_text(s):
    steps = []
    if s["filter_method"] == "Median Filter":
        steps.append(f"Median filter ({s['median_k']}x{s['median_k']})")
    elif s["filter_method"] == "Gaussian Filter":
        steps.append(f"Gaussian filter ({s['gauss_k']}x{s['gauss_k']})")
    elif s["filter_method"] == "Bilateral Filter":
        steps.append(f"Bilateral filter (sigma {s['bilateral_sigma']})")
    if s["sharpening"]:
        steps.append(f"Unsharp mask (amount {s['sharp_strength']}, radius {s['sharp_radius']})")
    if s["contrast"]:
        steps.append(f"CLAHE (clip {s['clahe_clip']})")
    if upscale_factor_value(s) > 1:
        steps.append(f"Bicubic upscaling {s['upscale_factor']}")
    return " -> ".join(steps) if steps else "No restoration applied"


# =====================================================================
# 4. QUALITY METRICS AND FILE HANDLING
# =====================================================================

def compute_metrics(reference, test):
    """PSNR and SSIM. Returns None unless both images have identical shape."""
    if reference is None or test is None or reference.shape != test.shape:
        return None
    if np.array_equal(reference, test):
        return float("inf"), 1.0
    psnr = peak_signal_noise_ratio(reference, test, data_range=255)
    try:
        ssim = structural_similarity(reference, test, channel_axis=2, data_range=255)
    except ValueError:
        return psnr, None
    return psnr, ssim


def fmt_psnr(value):
    if value is None:
        return "N/A"
    return "Identical" if np.isinf(value) else f"{value:.2f} dB"


def fmt_ssim(value):
    return "N/A" if value is None else f"{value:.4f}"


def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._") or "image"


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def inside_folder(path, folder):
    """True if `path` really lives directly inside `folder` (path validation)."""
    return path.resolve().parent == folder.resolve()


def save_original(source):
    """Store the untouched upload in images/. The name includes a content hash,
    so the same file is never saved twice and different files never overwrite."""
    name = f"{source['stem']}_{source['sha'][:8]}{source['ext']}"
    path = IMAGES_DIR / name
    if not inside_folder(path, IMAGES_DIR):
        raise ValueError("Unsafe file name.")
    if not path.exists():
        path.write_bytes(source["data"])
    return path


def list_images(folder):
    files = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def make_thumbnail(path_str, mtime, size=420):
    with Image.open(path_str) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size))
        return im.copy()


def save_result(result, source, exp_name, s, in_shape, metrics, notes):
    """Save the enhanced image to outputs/ and add a row to the experiment log.
    Returns (file_name, was_new). The same result is not saved twice."""
    signature = hashlib.sha1(result.tobytes()).hexdigest()[:10]
    saved_map = st.session_state.setdefault("saved_results", {})
    existing = saved_map.get(signature)
    if existing and (OUTPUTS_DIR / existing).exists():
        return existing, False

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{source['stem']}_enhanced_{stamp}_{signature[:6]}"
    path, n = OUTPUTS_DIR / f"{base}.png", 1
    while path.exists():  # never overwrite an existing file
        path = OUTPUTS_DIR / f"{base}_{n}.png"
        n += 1
    if not inside_folder(path, OUTPUTS_DIR):
        raise ValueError("Unsafe file name.")
    Image.fromarray(result).save(path, format="PNG")
    saved_map[signature] = path.name

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": exp_name or "Custom experiment",
        "source_image": source["name"],
        "degradation": degradation_text(s),
        "restoration": restoration_text(s),
        "input_size": f"{in_shape[1]}x{in_shape[0]}",
        "output_size": f"{result.shape[1]}x{result.shape[0]}",
        "psnr_db": "" if metrics is None else fmt_psnr(metrics[0]).replace(" dB", ""),
        "ssim": "" if metrics is None or metrics[1] is None else f"{metrics[1]:.4f}",
        "output_file": path.name,
        "observations": " ".join(notes.split()),
    }
    is_new_log = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if is_new_log:
            writer.writeheader()
        writer.writerow(row)
    return path.name, True


def read_log():
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# =====================================================================
# 5. PAGE STYLING
# =====================================================================

def _version_tuple():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# Newer Streamlit uses width="stretch"; older versions use use_container_width.
STRETCH = {"width": "stretch"} if _version_tuple() >= (1, 50) else {"use_container_width": True}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#14213D;--blue:#2563EB;--cyan:#06B6D4;--bg:#F4F7FB;--card:#FFFFFF;
--text:#172033;--muted:#64748B;--border:#E2E8F0;--ok:#16A34A;--err:#DC2626;}
html, body, .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label,
.stApp button, .stApp input, .stApp textarea, [data-testid="stMarkdownContainer"] *{
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
.block-container{max-width:1380px;padding-top:2rem;padding-bottom:3rem;}
footer{visibility:hidden;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--navy);}
[data-testid="stSidebar"] *{color:#E8EEFB;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.14);}
[data-testid="stSidebar"] .stButton > button{
  justify-content:flex-start;background:transparent;border:1px solid transparent;
  border-radius:10px;min-height:42px;font-weight:550;}
[data-testid="stSidebar"] .stButton > button:hover{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.18);}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{
  background:var(--blue);border-color:var(--blue);color:#fff;}
.brand{display:flex;align-items:center;gap:12px;margin:4px 0 2px;}
.brand img{width:42px;height:42px;border-radius:10px;}
.brand-name{font-size:1.2rem;font-weight:750;color:#fff;line-height:1.15;}
.brand-tag{font-size:.78rem;color:#A9BAD9;line-height:1.3;}
.side-note{font-size:.82rem;color:#A9BAD9;line-height:1.5;}

/* Page header and hero */
.page-head{display:flex;align-items:center;gap:14px;margin-bottom:6px;}
.page-head img{width:46px;height:46px;border-radius:11px;}
.page-title{font-size:1.65rem;font-weight:750;color:var(--navy);line-height:1.2;margin:0;}
.page-sub{font-size:.95rem;color:var(--muted);margin:2px 0 0;}
.hero{background:var(--card);border:1px solid var(--border);border-top:4px solid var(--cyan);
  border-radius:18px;padding:34px 38px;margin-bottom:22px;}
.hero img.hero-logo{height:56px;margin-bottom:18px;}
.hero h1{font-size:2.3rem;font-weight:800;color:var(--navy);margin:0 0 10px;line-height:1.15;letter-spacing:-.01em;}
.hero p{font-size:1.05rem;color:var(--muted);margin:0;max-width:640px;line-height:1.6;}

/* Sections and cards */
.sec{display:flex;align-items:center;gap:12px;margin:30px 0 14px;}
.sec-ic{width:38px;height:38px;border-radius:10px;background:#E8F0FE;display:flex;
  align-items:center;justify-content:center;flex:0 0 38px;}
.sec-title{font-size:1.2rem;font-weight:700;color:var(--navy);line-height:1.25;}
.sec-sub{font-size:.9rem;color:var(--muted);line-height:1.4;}
.fcard{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;height:100%;
  transition:border-color .15s ease, box-shadow .15s ease;}
.fcard:hover{border-color:var(--blue);box-shadow:0 4px 14px rgba(37,99,235,.10);}
.fcard .ic{width:40px;height:40px;border-radius:10px;background:#E8F0FE;display:flex;
  align-items:center;justify-content:center;margin-bottom:12px;}
.fcard-title{font-size:1rem;font-weight:650;color:var(--navy);margin-bottom:4px;}
.fcard-text{font-size:.88rem;color:var(--muted);line-height:1.5;}
.empty{background:var(--card);border:1px dashed #CBD5E1;border-radius:14px;padding:28px;text-align:center;color:var(--muted);}
.empty b{display:block;color:var(--navy);font-size:1rem;margin-bottom:4px;}
.chip{display:inline-block;background:#E8F0FE;color:#1D4ED8;border-radius:999px;padding:3px 11px;
  font-size:.8rem;font-weight:600;margin:0 6px 6px 0;}
.ie-icon{vertical-align:-4px;}

/* Native widgets */
div.stButton > button, div.stDownloadButton > button{border-radius:10px;min-height:44px;font-weight:600;}
div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:var(--blue);border-color:var(--blue);color:#fff;}
div.stButton > button[kind="primary"]:hover{background:#1D4ED8;border-color:#1D4ED8;}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 16px;}
[data-testid="stMetricLabel"] p{color:var(--muted);font-size:.82rem;font-weight:500;}
[data-testid="stMetricValue"]{color:var(--navy);font-weight:750;}
[data-testid="stFileUploader"] section{background:var(--card);border:2px dashed #93C5FD;border-radius:14px;}
[data-testid="stAlert"]{border-radius:12px;}
[data-testid="stImage"] img{border-radius:10px;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def logo_mark_uri():
    return file_data_uri(MARK_PATH, "image/svg+xml", MARK_SVG)


def logo_full_uri():
    return file_data_uri(LOGO_PATH, "image/svg+xml", LOGO_SVG)


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-head"><img src="{logo_mark_uri()}" alt="ImageEnhance logo">'
        f'<div><p class="page-title">{title}</p><p class="page-sub">{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def section_header(icon, title, subtitle=""):
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec"><div class="sec-ic">{icon_html(icon)}</div>'
        f'<div><div class="sec-title">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def feature_card(icon, title, text):
    return (
        f'<div class="fcard"><div class="ic">{icon_html(icon)}</div>'
        f'<div class="fcard-title">{title}</div><div class="fcard-text">{text}</div></div>'
    )


def empty_state(title, text):
    st.markdown(f'<div class="empty"><b>{title}</b>{text}</div>', unsafe_allow_html=True)


def footer():
    st.divider()
    st.caption(
        f"{APP_NAME} | {PROJECT_TITLE} | Built with Python, Streamlit, OpenCV, "
        "NumPy, Pillow and scikit-image"
    )


def comparison_slider(before, after, before_label, after_label, height=520):
    """Drag-to-compare slider (plain HTML/JS, no extra packages)."""
    def uri(arr):
        img = Image.fromarray(arr)
        img.thumbnail((1100, 1100))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    html = f"""
    <style>
      body{{margin:0;font-family:'Segoe UI',Arial,sans-serif;}}
      .wrap{{position:relative;height:{height - 12}px;border-radius:12px;overflow:hidden;background:#0F172A;}}
      .wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;}}
      #top{{clip-path:inset(0 50% 0 0);}}
      #line{{position:absolute;top:0;bottom:0;left:50%;width:3px;margin-left:-1.5px;background:#fff;
             box-shadow:0 0 6px rgba(0,0,0,.5);pointer-events:none;}}
      .tag{{position:absolute;top:10px;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;
            background:rgba(20,33,61,.85);color:#fff;pointer-events:none;}}
      input[type=range]{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0;}}
    </style>
    <div class="wrap">
      <img src="{uri(after)}" alt="{after_label}">
      <img id="top" src="{uri(before)}" alt="{before_label}">
      <div id="line"></div>
      <span class="tag" style="left:10px">{before_label}</span>
      <span class="tag" style="right:10px">{after_label}</span>
      <input id="r" type="range" min="0" max="100" value="50" aria-label="Comparison slider">
    </div>
    <script>
      const r=document.getElementById('r'), t=document.getElementById('top'), l=document.getElementById('line');
      r.addEventListener('input',()=>{{t.style.clipPath='inset(0 '+(100-r.value)+'% 0 0)';l.style.left=r.value+'%';}});
    </script>
    """
    components.html(html, height=height)


# =====================================================================
# 6. PAGE LAYOUTS
# =====================================================================

# ---------- session state helpers ----------

def init_settings():
    """Keep control values when the user moves between pages."""
    saved = st.session_state.get("_saved_settings", {})
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = saved.get(key, default)
    st.session_state.setdefault("enhanced", False)
    st.session_state.setdefault("exp_name", "Custom experiment")


def reset_settings():
    for key, default in DEFAULTS.items():
        st.session_state[key] = default
    st.session_state["enhanced"] = False
    st.session_state["preset_choice"] = NO_PRESET
    st.session_state["exp_name"] = "Custom experiment"


def load_preset():
    choice = st.session_state.get("preset_choice", NO_PRESET)
    if choice in PRESETS:
        for key, default in DEFAULTS.items():
            st.session_state[key] = PRESETS[choice].get(key, default)
        st.session_state["exp_name"] = choice
        st.session_state["enhanced"] = False


def start_enhancing():
    st.session_state["enhanced"] = True


def go_to(page):
    st.session_state["page"] = page


def current_settings():
    return {key: st.session_state[key] for key in DEFAULTS}


# ---------- sidebar ----------

NAV_ITEMS = [
    ("workspace", "Image Workspace", ":material/home:"),
    ("saved", "Saved Images", ":material/folder:"),
    ("history", "Experiment History", ":material/science:"),
    ("help", "Help & Limitations", ":material/help:"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><img src="{logo_mark_uri()}" alt="">'
            f'<div><div class="brand-name">{APP_NAME}</div>'
            f'<div class="brand-tag">{TAGLINE}</div></div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        for page, label, icon in NAV_ITEMS:
            active = st.session_state["page"] == page
            st.button(
                label, icon=icon, key=f"nav_{page}", on_click=go_to, args=(page,),
                type="primary" if active else "secondary", **STRETCH,
            )
        st.divider()
        n_orig, n_out = len(list_images(IMAGES_DIR)), len(list_images(OUTPUTS_DIR))
        st.markdown(
            f'<div class="side-note">Original uploads: <b>{n_orig}</b><br>'
            f'Saved outputs: <b>{n_out}</b><br>Experiment log rows: <b>{len(read_log())}</b></div>',
            unsafe_allow_html=True,
        )


# ---------- workspace: upload ----------

def render_hero():
    st.markdown(
        f'<div class="hero"><img class="hero-logo" src="{logo_full_uri()}" alt="ImageEnhance">'
        f"<h1>{TAGLINE}</h1>"
        "<p>Transform noisy, blurry, and low-resolution images using practical "
        "digital image processing techniques.</p></div>",
        unsafe_allow_html=True,
    )


def render_overview_cards():
    section_header("sparkle", "What You Can Do", "Five classical techniques, each applied to the same input for fair comparison.")
    cards = [
        ("restore", "Image Restoration", "Recover a cleaner image from a degraded copy."),
        ("noise", "Noise Reduction", "Remove Gaussian and salt-and-pepper noise with filters."),
        ("sharpen", "Image Sharpening", "Bring back edge definition with unsharp masking."),
        ("contrast", "Contrast Enhancement", "Balance lighting with CLAHE on the lightness channel."),
        ("upscale", "Resolution Upscaling", "Enlarge images with bicubic interpolation."),
    ]
    cols = st.columns(5)
    for col, (icon, title, text) in zip(cols, cards):
        col.markdown(feature_card(icon, title, text), unsafe_allow_html=True)


def handle_upload():
    """Show the uploader and keep the current image in session state."""
    section_header("upload", "Upload Your Image", "Upload an image to begin your restoration workflow.")
    uploaded = st.file_uploader(
        "Choose a JPG, PNG, BMP, or TIFF image", type=ALLOWED_TYPES,
        key="main_upload", label_visibility="collapsed",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        sha = hashlib.sha1(data).hexdigest()
        current = st.session_state.get("source")
        if current is None or current["sha"] != sha:
            try:
                decode_image(data)  # make sure it really is a readable image
            except Exception as exc:
                st.error(f"This file could not be read as an image ({exc}). Try a JPG or PNG file.")
                return None
            suffix = Path(uploaded.name).suffix.lower()
            st.session_state["source"] = {
                "name": uploaded.name,
                "stem": safe_name(Path(uploaded.name).stem),
                "ext": suffix if suffix in ALLOWED_SUFFIXES else ".png",
                "data": data,
                "sha": sha,
            }
            st.session_state["enhanced"] = False
    source = st.session_state.get("source")
    if source is None:
        return None

    try:
        saved_path = save_original(source)
    except Exception as exc:
        st.error(f"Could not store the original image: {exc}")
        return None
    source["saved_as"] = saved_path.name

    if uploaded is None:
        c1, c2 = st.columns([4, 1])
        c1.info(f"Using the image already loaded: **{source['name']}**. Upload another file to replace it.")
        if c2.button("Clear image", icon=":material/close:", **STRETCH,
                     help="Removes the image from the workspace only. Your saved files are not deleted."):
            st.session_state["source"] = None
            st.session_state["enhanced"] = False
            st.rerun()
    return source


def render_image_info(source, work, full_size, resized):
    section_header("image", "Original Image", "File details and storage location.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", source["name"] if len(source["name"]) < 22 else source["name"][:19] + "...")
    c2.metric("Dimensions", f"{full_size[0]} x {full_size[1]} px")
    c3.metric("File size", f"{len(source['data']) / 1024:,.0f} KB")
    c4.metric("Stored as", f"images/{source['saved_as']}" if len(source["saved_as"]) < 20 else "images/ folder")
    if resized:
        st.caption(
            f"Large image: processing uses a {work.shape[1]} x {work.shape[0]} px copy so the app stays fast. "
            "Your stored original is unchanged."
        )
    with st.expander("Preview original image"):
        st.image(work, **STRETCH)


# ---------- workspace: controls ----------

def render_controls():
    section_header("sliders", "Restoration Controls",
                   "Choose a problem to simulate, then pick the methods that should fix it.")
    with st.container(border=True):
        p1, p2 = st.columns([3, 1], vertical_alignment="bottom")
        p1.selectbox("Quick experiment preset", [NO_PRESET] + list(PRESETS), key="preset_choice",
                     help="Loads settings for one of the four project experiments.")
        p2.button("Load Preset", icon=":material/playlist_add_check:", on_click=load_preset, **STRETCH)

        t1, t2, t3 = st.tabs(["Image Problem", "Noise & Sharpness", "Contrast & Resolution"])

        with t1:
            st.selectbox("Simulate image problem", DEGRADATIONS, key="degradation",
                         help="Adds a known problem to your image so the restoration can be measured against the original.")
            mode = st.session_state["degradation"]
            if mode == "Gaussian Noise":
                st.slider("Gaussian noise strength", 5, 50, key="noise_level",
                          help="Standard deviation of the added noise. Higher means grainier.")
            elif mode == "Salt-and-Pepper Noise":
                st.slider("Noise amount (fraction of pixels)", 0.01, 0.15, step=0.01, key="sp_amount",
                          help="Share of pixels turned pure black or pure white.")
            elif mode == "Gaussian Blur":
                st.select_slider("Blur strength (kernel size)", options=[3, 5, 7, 9, 11, 13, 15], key="blur_strength",
                                 help="Larger kernels blur more.")
            elif mode == "Low Resolution":
                st.select_slider("Downscaling factor", options=[2, 3, 4], key="down_factor",
                                 help="The image is shrunk by this factor. Choose the same upscale factor to return to the original size.")
                st.caption("Tip: set Resolution Upscaling to the same factor so PSNR and SSIM can be calculated.")

        with t2:
            st.selectbox("Noise Reduction filter", FILTERS, key="filter_method",
                         help="Median suits salt-and-pepper noise. Gaussian smooths grain but softens edges. "
                              "Bilateral smooths while keeping edges.")
            method = st.session_state["filter_method"]
            if method == "Median Filter":
                st.select_slider("Median kernel size", options=[3, 5, 7, 9], key="median_k")
            elif method == "Gaussian Filter":
                st.select_slider("Gaussian kernel size", options=[3, 5, 7, 9], key="gauss_k")
            elif method == "Bilateral Filter":
                st.slider("Bilateral colour sigma", 25, 150, key="bilateral_sigma",
                          help="Higher values smooth more strongly across colour differences.")
            st.checkbox("Apply sharpening (unsharp mask)", key="sharpening",
                        help="Subtracts a blurred copy to boost edges. Too much creates halos and amplifies noise.")
            if st.session_state["sharpening"]:
                s1, s2 = st.columns(2)
                s1.slider("Sharpening strength", 0.0, 2.0, step=0.1, key="sharp_strength")
                s2.slider("Sharpening radius", 0.5, 5.0, step=0.5, key="sharp_radius")

        with t3:
            st.checkbox("Apply CLAHE contrast enhancement", key="contrast",
                        help="Adaptive histogram equalisation on the lightness channel. Colours are preserved.")
            if st.session_state["contrast"]:
                st.slider("CLAHE clip limit", 1.0, 6.0, step=0.5, key="clahe_clip",
                          help="Higher values give stronger contrast but can exaggerate noise.")
            st.selectbox("Resolution Upscaling (bicubic)", UPSCALES, key="upscale_factor",
                         help="Increases pixel dimensions by interpolation. It does not recover real missing detail.")

        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("Enhance Image", icon=":material/auto_fix_high:", type="primary",
                  on_click=start_enhancing, **STRETCH)
        b2.button("Reset Settings", icon=":material/restart_alt:", on_click=reset_settings, **STRETCH)


# ---------- workspace: results ----------

def render_comparison(s, reference, degraded, result):
    section_header("compare", "Before & After Comparison", "Your Image, Improved")
    st.markdown(
        f'<span class="chip">Problem: {degradation_text(s)}</span>'
        f'<span class="chip">Restoration: {restoration_text(s)}</span>',
        unsafe_allow_html=True,
    )
    view = st.radio("View", ["Side by side", "Comparison slider"], horizontal=True,
                    key="view_mode", label_visibility="collapsed")

    if view == "Comparison slider":
        try:
            comparison_slider(degraded, result, "Input", "Enhanced")
            st.caption(f"Input: {degraded.shape[1]} x {degraded.shape[0]} px | "
                       f"Enhanced: {result.shape[1]} x {result.shape[0]} px. Drag across the image to compare.")
            return
        except Exception:
            st.warning("The comparison slider could not be shown, so the side-by-side view is used instead.")

    if s["degradation"] == "None":
        panels = [("Original", reference), ("Enhanced Output", result)]
    else:
        panels = [("Original", reference), ("Degraded Input", degraded), ("Enhanced Output", result)]
    for col, (label, img) in zip(st.columns(len(panels), gap="medium"), panels):
        with col:
            st.markdown(f"**{label}**")
            st.image(img, **STRETCH)
            st.caption(f"{img.shape[1]} x {img.shape[0]} px")
    if degraded.shape != reference.shape or result.shape != reference.shape:
        st.caption("Images are drawn at the same width for viewing. Read the captions for true pixel sizes.")


def render_quality(s, reference, degraded, result):
    """Returns (metrics, reference_array). metrics = (psnr, ssim) or None."""
    section_header("chart", "Image Quality Analysis",
                   "Evaluate image similarity using PSNR and SSIM when a valid clean reference image is available.")
    ref_file = st.file_uploader("Optional: upload a clean reference image of the same scene",
                                type=ALLOWED_TYPES, key="ref_upload")
    ref_arr, ref_label = None, ""
    if ref_file is not None:
        try:
            ref_arr = decode_image(ref_file.getvalue())[0]
            ref_label = "your uploaded reference"
        except Exception as exc:
            st.error(f"The reference image could not be read: {exc}")
    elif s["degradation"] != "None":
        ref_arr = reference
        ref_label = "the original image (the clean source of the simulated problem)"

    c1, c2, c3 = st.columns(3)
    c1.metric("Original size", f"{reference.shape[1]} x {reference.shape[0]}")
    c2.metric("Input size", f"{degraded.shape[1]} x {degraded.shape[0]}")
    c3.metric("Output size", f"{result.shape[1]} x {result.shape[0]}")
    p1, p2, p3 = st.columns(3)
    p1.metric("Pixels before", f"{degraded.shape[0] * degraded.shape[1]:,}")
    p2.metric("Pixels after", f"{result.shape[0] * result.shape[1]:,}")
    p3.metric("Pixel change", f"{result.shape[0] * result.shape[1] / (degraded.shape[0] * degraded.shape[1]):.2f}x")

    metrics = None
    if ref_arr is None:
        st.info("PSNR and SSIM need a clean reference. Choose a simulated problem so your original becomes "
                "the reference, or upload a clean image of the same scene.")
    elif ref_arr.shape != result.shape:
        st.warning(
            f"PSNR and SSIM are not calculated because the reference is {ref_arr.shape[1]} x {ref_arr.shape[0]} px "
            f"and the output is {result.shape[1]} x {result.shape[0]} px. Both images must have identical "
            "dimensions and show the same aligned scene."
            + (" For the low-resolution experiment, set Resolution Upscaling to the downscaling factor."
               if s["degradation"] == "Low Resolution" else ""))
    else:
        metrics = compute_metrics(ref_arr, result)
        base = compute_metrics(ref_arr, degraded)
        m1, m2 = st.columns(2)
        psnr_delta = ssim_delta = None
        if base is not None and not np.isinf(metrics[0]) and not np.isinf(base[0]):
            psnr_delta = f"{metrics[0] - base[0]:+.2f} dB vs input"
            if metrics[1] is not None and base[1] is not None:
                ssim_delta = f"{metrics[1] - base[1]:+.4f} vs input"
        m1.metric("PSNR (higher is better)", fmt_psnr(metrics[0]), delta=psnr_delta)
        m2.metric("SSIM (closer to 1 is better)", fmt_ssim(metrics[1]), delta=ssim_delta)
        st.caption(f"Measured against {ref_label}. A positive change means the output is closer to the reference than the input was.")
        if ref_file is not None:
            st.caption("Make sure the uploaded reference shows exactly the same scene, aligned with your image.")

    if upscale_factor_value(s) > 1:
        st.info("Bicubic interpolation increases the pixel count but cannot recreate detail that was never captured. "
                "It is not true super-resolution.")
    return metrics, ref_arr


def render_filter_comparison(s, degraded, ref_arr):
    with st.expander("Fair Filter Comparison (same degraded input)"):
        st.caption("Each filter below is applied to the same degraded image, never to an already processed result.")
        if not st.toggle("Run comparison", key="run_compare"):
            return
        variants = [("No filter", degraded)] + [(m, apply_noise_filter(degraded, m, s)) for m in FILTERS[1:]]
        for col, (name, img) in zip(st.columns(len(variants)), variants):
            with col:
                st.markdown(f"**{name}**")
                st.image(img, **STRETCH)
                m = compute_metrics(ref_arr, img)
                if m is not None:
                    st.caption(f"PSNR {fmt_psnr(m[0])} | SSIM {fmt_ssim(m[1])}")
                else:
                    st.caption("No valid reference for these dimensions")


def render_export(s, source, degraded, result, metrics):
    section_header("download", "Export Enhanced Image", "Download a copy, or save it to your project folders with notes.")
    with st.container(border=True):
        st.text_input("Experiment name", key="exp_name")
        notes = st.text_area("Observations (saved in the experiment log)", key="obs_text", height=90,
                             placeholder="Example: Median filtering removed most salt-and-pepper noise but softened fine texture.")
        d_col, s_col = st.columns(2)
        d_col.download_button(
            "Download Enhanced Image", data=png_bytes(result), icon=":material/download:",
            file_name=f"{source['stem']}_enhanced.png", mime="image/png", **STRETCH,
        )
        if s_col.button("Save to Project", icon=":material/save:", type="primary", **STRETCH):
            try:
                name, was_new = save_result(result, source, st.session_state["exp_name"], s,
                                            degraded.shape, metrics, notes)
                if was_new:
                    st.success(f"Saved as outputs/{name} and added to the experiment log.")
                else:
                    st.info(f"This exact result is already saved as outputs/{name}. Nothing was duplicated.")
            except Exception as exc:
                st.error(f"Could not save the image: {exc}")


def render_workspace():
    init_settings()
    source = handle_upload() if st.session_state.get("source") is None else None
    if st.session_state.get("source") is None:
        # No image yet: show the welcome hero above the uploader.
        pass
    return source


# ---------- workspace: main ----------

def page_workspace():
    init_settings()
    has_image = st.session_state.get("source") is not None
    if has_image:
        page_header("Image Workspace", "Upload, restore, compare, and export your image.")
    else:
        render_hero()

    source = handle_upload()
    if source is None:
        render_overview_cards()
        return

    try:
        work, full_size, resized = decode_image(source["data"])
    except Exception as exc:
        st.error(f"The image could not be opened: {exc}")
        return

    render_image_info(source, work, full_size, resized)
    render_controls()

    if not st.session_state["enhanced"]:
        st.info("Adjust the controls above, then select **Enhance Image** to see your results.")
        st.session_state["_saved_settings"] = current_settings()
        return

    s = current_settings()
    try:
        with st.spinner("Enhancing image..."):
            degraded, reference = degrade_image(work, s)
            result = restore_image(degraded, s)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.session_state["_saved_settings"] = s
        return

    render_comparison(s, reference, degraded, result)
    metrics, ref_arr = render_quality(s, reference, degraded, result)
    if s["degradation"] != "None" or ref_arr is not None:
        render_filter_comparison(s, degraded, ref_arr)
    render_export(s, source, degraded, result, metrics)
    st.session_state["_saved_settings"] = s


# ---------- other pages ----------

def gallery(folder, key_prefix, empty_title, empty_text):
    files = list_images(folder)
    if not files:
        empty_state(empty_title, empty_text)
        return
    st.caption(f"{len(files)} image(s) in {folder.name}/, newest first.")
    show_all = len(files) > 12 and st.toggle("Show all images", key=f"{key_prefix}_all")
    shown = files if show_all else files[:12]
    cols = st.columns(3, gap="medium")
    for i, path in enumerate(shown):
        with cols[i % 3]:
            with st.container(border=True):
                try:
                    st.image(make_thumbnail(str(path), path.stat().st_mtime), **STRETCH)
                except Exception:
                    st.warning("Preview unavailable")
                st.caption(path.name)
                st.download_button("Download", data=path.read_bytes(), file_name=path.name,
                                   icon=":material/download:", key=f"dl_{key_prefix}_{path.name}",
                                   mime="image/png" if path.suffix.lower() == ".png" else "application/octet-stream",
                                   **STRETCH)
    with st.expander("Preview a saved image at full size"):
        choice = st.selectbox("Image", [p.name for p in files], key=f"{key_prefix}_preview")
        chosen = folder / choice
        if chosen.exists() and inside_folder(chosen, folder):
            with Image.open(chosen) as im:
                st.image(ImageOps.exif_transpose(im).convert("RGB"), **STRETCH)
                st.caption(f"{im.size[0]} x {im.size[1]} px")


def page_saved():
    page_header("Saved Images", "Enhanced results and stored originals in your project folders.")
    section_header("folder", "Saved Images", "Files are never deleted by this app.")
    t1, t2 = st.tabs(["Enhanced Outputs", "Original Uploads"])
    with t1:
        gallery(OUTPUTS_DIR, "out", "No saved results yet",
                "Enhance an image in the workspace, then select Save to Project.")
    with t2:
        gallery(IMAGES_DIR, "orig", "No uploaded originals yet",
                "Images you upload are stored here automatically.")


def page_history():
    page_header("Experiment History", "Every saved result is logged with its settings and measurements.")
    rows = read_log()
    if not rows:
        empty_state("No experiments logged yet",
                    "Save a result from the workspace to start your experiment log.")
        return
    section_header("flask", "Experiment History", f"{len(rows)} logged experiment(s), newest first.")
    cols = st.columns(3, gap="medium")
    for i, row in enumerate(reversed(rows[-12:])):
        with cols[i % 3]:
            with st.container(border=True):
                out = OUTPUTS_DIR / row.get("output_file", "")
                if row.get("output_file") and out.exists() and inside_folder(out, OUTPUTS_DIR):
                    st.image(make_thumbnail(str(out), out.stat().st_mtime), **STRETCH)
                st.markdown(f"**{row['experiment']}**")
                st.caption(row["timestamp"])
                st.markdown(f"Problem: {row['degradation']}  \nMethods: {row['restoration']}")
                psnr = row["psnr_db"] or "N/A"
                ssim = row["ssim"] or "N/A"
                st.markdown(f"PSNR: **{psnr}** | SSIM: **{ssim}**")
                if row["observations"]:
                    st.caption(row["observations"])
    with st.expander("Full log table"):
        st.dataframe(rows, **STRETCH)
    st.download_button("Download experiment log (CSV)", data=LOG_FILE.read_bytes(),
                       file_name="experiment_log.csv", mime="text/csv", icon=":material/download:")


def page_help():
    page_header("Help & Limitations", "What each technique does and where it falls short.")
    section_header("help", "Techniques", "Choose the tool that matches the problem.")
    table = (
        "| Technique | Best for | Limitation |\n|---|---|---|\n"
        "| Median filter | Salt-and-pepper noise | Softens fine texture at large kernel sizes |\n"
        "| Gaussian filter | Light Gaussian noise | Blurs edges along with the noise |\n"
        "| Bilateral filter | Noise while keeping edges | Slower; can give a flat, painted look |\n"
        "| Unsharp mask | Mild blur | Cannot restore lost detail; creates halos and boosts noise if too strong |\n"
        "| CLAHE | Flat or unevenly lit images | Can exaggerate noise in smooth regions |\n"
        "| Bicubic upscaling | Enlarging an image | Adds pixels, not new detail; it is not super-resolution |"
    )
    st.markdown(table)
    section_header("chart", "Reading the quality scores")
    st.markdown(
        "- **PSNR** compares pixel values in decibels. Higher is closer to the reference.\n"
        "- **SSIM** compares structure and runs from 0 to 1. Closer to 1 is more similar.\n"
        "- Scores are only valid when the reference shows the same aligned scene and has exactly the same dimensions.\n"
        "- A better score does not always look better to the eye. Check the images as well."
    )
    section_header("folder", "Where your files go")
    st.markdown(
        "- `images/` stores the untouched upload (one copy per unique file).\n"
        "- `outputs/` stores enhanced images only when you select **Save to Project**.\n"
        "- `experiments/experiment_log.csv` records the settings, notes and scores of each saved result."
    )


# =====================================================================
# 7. MAIN
# =====================================================================

def main():
    problems = prepare_folders()
    ensure_assets()

    try:
        icon = Image.open(FAVICON_PATH) if FAVICON_PATH.exists() else "🖼️"
    except Exception:
        icon = "🖼️"
    st.set_page_config(page_title=f"{APP_NAME} | {PROJECT_TITLE}", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    if problems:
        st.error("A project folder could not be created because a file with the same name exists.")
        for p in problems:
            st.markdown(f"- `{p}` is a file. Rename it (for example to `{p.name}_old`), then reload this page.")
        st.stop()

    st.session_state.setdefault("page", "workspace")
    render_sidebar()

    page = st.session_state["page"]
    if page == "saved":
        page_saved()
    elif page == "history":
        page_history()
    elif page == "help":
        page_help()
    else:
        page_workspace()
    footer()


main()
