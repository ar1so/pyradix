# pyradix in a container.
#
#   docker build -t pyradix .
#   docker run --rm -v "$PWD:/data" pyradix extract image.nii.gz mask.nii.gz --out features.csv
#
# Stage 1 compiles the Rust engine into a wheel; stage 2 keeps only Python
# and the installed wheel.

FROM python:3.12-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config gfortran libopenblas-dev patchelf curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.14,<1.16"
WORKDIR /src
COPY . .
RUN maturin build --release --out /dist

FROM python:3.12-slim
LABEL org.opencontainers.image.title="pyradix" \
      org.opencontainers.image.description="Radiomics feature extraction: IBSI and PyRadiomics 3.1.0 modes" \
      org.opencontainers.image.source="https://github.com/ar1so/pyradix" \
      org.opencontainers.image.licenses="BSD-3-Clause"
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgfortran5 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl \
 && useradd --create-home --uid 1000 radix
USER radix
WORKDIR /data
ENTRYPOINT ["pyradix"]
CMD ["--help"]
