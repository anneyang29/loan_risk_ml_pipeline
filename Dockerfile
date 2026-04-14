# ============================================================
# Loan Risk ML Pipeline — Dockerfile
# ============================================================
# Base image: python:3.11-slim-bookworm
#   - Slim Debian 12 image; small footprint, no conda overhead.
#   - PySpark requires Java; we install OpenJDK 17 via apt.
#   - Chinese font rendering in matplotlib requires CJK fonts.
#     fonts-noto-cjk  : installs as "Noto Sans CJK TC/SC/HK/JP"
#     fonts-wqy-zenhei: "WenQuanYi Zen Hei" — secondary CJK fallback
#     fontconfig      : provides fc-cache to register fonts system-wide
# ============================================================

FROM python:3.11-slim-bookworm

# ── Labels ────────────────────────────────────────────────────
LABEL maintainer="loan-risk-ml-pipeline"
LABEL description="Four-phase credit risk ML pipeline with challenger evaluation and governance reporting"

# ── System dependencies ───────────────────────────────────────
# - default-jdk-headless : OpenJDK 17 required by PySpark local mode
# - fonts-noto-cjk       : Noto Sans CJK TC/SC/HK/JP/KR — primary CJK font
#                          matplotlib sees these as "Noto Sans CJK TC" etc.
# - fonts-wqy-zenhei     : WenQuanYi Zen Hei — secondary CJK fallback
# - fontconfig           : provides fc-cache binary; must be explicit on slim images
# - libgomp1             : OpenMP runtime required by XGBoost parallel training
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless \
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    fontconfig \
    libgomp1 \
    && fc-cache -fv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── JAVA_HOME — PySpark needs this ────────────────────────────
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ── Python environment ─────────────────────────────────────────
RUN pip install --upgrade pip --no-cache-dir

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer-cache friendly) ────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────
# .dockerignore excludes data/, datamart/, model_bank/, __pycache__/, etc.
COPY . .

# ── Rebuild matplotlib font cache ─────────────────────────────
# Must run AFTER both:
#   1. pip installs matplotlib  (done above)
#   2. system fonts are installed via apt  (done above)
# Deletes the stale cache file first so matplotlib re-scans /usr/share/fonts
# and picks up fonts-noto-cjk ("Noto Sans CJK TC" etc.) and fonts-wqy-zenhei.
RUN python -W ignore -c "\
import matplotlib.font_manager as fm; \
import pathlib; \
cache = pathlib.Path(fm.get_cachedir()); \
[p.unlink() for p in cache.glob('fontlist-*.json')]; \
fm._load_fontmanager(try_read_cache=False); \
names = sorted(set(f.name for f in fm.fontManager.ttflist)); \
cjk = [n for n in names if any(k in n for k in ['CJK','WenQuan','Noto Sans CJK'])]; \
print('CJK fonts visible to matplotlib:', cjk); \
"

# ── Spark scratch directories ─────────────────────────────────
ENV SPARK_LOCAL_DIRS=/tmp/spark
RUN mkdir -p /tmp/spark

# ── Default command ────────────────────────────────────────────
CMD ["python", "main.py", "--report-only"]
