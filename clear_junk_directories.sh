#!/bin/bash

find /datadisk/whiteboxing/benchmarks2/*/*/ -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;
