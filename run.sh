#!/bin/sh

docker compose \
    -f docker-compose.yaml \
    -f docker-compose.with-sound.yaml \
    up \
    "$@"
