#!/bin/bash

find /datadisk/whiteboxing/2020Ada1/sens/*/*/ -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;
