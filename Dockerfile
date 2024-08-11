#######################################
FROM python:3.10-bookworm AS base

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        libsndfile1-dev && \
    apt-get clean

#######################################
FROM base AS build

RUN python3 -m venv /tmp/.hatch-venv
RUN /tmp/.hatch-venv/bin/pip install hatch

WORKDIR /build

COPY ./README.md /build/
COPY ./pyproject.toml /build/
COPY ./src/ /build/src/

RUN /tmp/.hatch-venv/bin/hatch build

#######################################
FROM base as runtime

COPY --from=build /build/dist/*.whl /build/dist/

RUN python3 -m venv /app/.venv
RUN /app/.venv/bin/pip install /build/dist/*.whl && rm -rf /build

ENTRYPOINT [ "/app/.venv/bin/localagi" ]
