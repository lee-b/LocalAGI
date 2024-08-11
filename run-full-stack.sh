#!/bin/bash

docker compose \
    -f docker-compose.yaml \
    -f docker-compose.with-localai.yaml \
    -f docker-compose.with-sound.yaml \
    up \
    "$@"
