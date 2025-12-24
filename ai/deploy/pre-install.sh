#!/bin/sh

echo "Installing linux dependencies"

apt-get update

apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential

apt-get install -y curl

apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
