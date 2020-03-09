#!/bin/bash

find /datadisk/whiteboxing/2020/*/*/*/ -maxdepth 1 -mindepth 1 -type d -exec rm -rf '{}' \;
