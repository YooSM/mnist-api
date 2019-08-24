#!/usr/bin/env bash
set -e

docker build \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t ${DOCKER_IMAGE} .

docker push ${DOCKER_IMAGE}
