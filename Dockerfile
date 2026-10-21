# syntax=docker/dockerfile:1
# DerivX FIC analytics: the Streamlit app in a slim, non-root image.
#
#   docker build -t derivx .                      # the runtime image (last stage)
#   docker run --rm -p 8501:8501 derivx           # then open http://localhost:8501
#   docker build --target test -t derivx-test .   # runtime layout + the UI smoke tests, run while building
#
# Settings come from .streamlit/config.toml (XSRF and CORS protection on) and
# the STREAMLIT_* variables below; override the port with -e STREAMLIT_SERVER_PORT=...
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Third-party packages first, so a code change reuses this layer. The lock pins
# the resolution CI tested for this Python (its header has the command that
# regenerates it); pyproject.toml holds the supported ranges.
COPY requirements-lock.txt ./
RUN python -m pip install -r requirements-lock.txt

# The package, editable so it finds its sample data in /app/data/examples.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN python -m pip install --no-deps -e . \
    && useradd --create-home --uid 10001 derivx
COPY .streamlit/config.toml .streamlit/config.toml
COPY data/examples/ data/examples/

# Recorded in run manifests (there is no .git in the image): --build-arg GIT_SHA=$(git rev-parse HEAD)
ARG GIT_SHA=unknown
ENV DERIVX_GIT_SHA=${GIT_SHA}

USER derivx

# --target test: the same image plus pytest, running the UI tests (AppTest for
# every mode and Rates tab, offline; QuantLib checks skip without QuantLib) and
# a live server on the image's config, so a broken image layout fails the build.
FROM app AS test
COPY --chown=derivx tests/ tests/
RUN python -m pip install --user "pytest>=8,<10" \
    && python -m pytest -q -p no:cacheprovider -W error::DeprecationWarning \
        "tests/test_deploy.py::test_committed_streamlit_config_keeps_protection_on_and_uses_known_options" \
        "tests/test_deploy.py::test_a_server_on_the_committed_config_refuses_tokenless_uploads_and_foreign_origins" \
        tests/test_pricer_page.py tests/test_fx_page.py tests/test_analytics_lab.py tests/test_walkforward_tab.py \
        tests/test_ui_state.py tests/test_plot_limits.py tests/test_swaps.py tests/test_capfloor.py \
        tests/test_bonds.py tests/test_scenarios.py

FROM app AS runtime

# DERIVX_CACHE_DIR: a writable place for downloaded market data (the app user
# cannot write under /app, which root owns).
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=type \
    DERIVX_CACHE_DIR=/home/derivx/.cache/derivx

EXPOSE 8501

# Healthy only when Streamlit answers 200 on its health endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('STREAMLIT_SERVER_PORT', '8501'), timeout=4)"]

CMD ["streamlit", "run", "src/derivx/ui/streamlit_app.py"]
