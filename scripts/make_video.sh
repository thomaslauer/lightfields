#!/bin/sh

ffmpeg -framerate 8 -pattern_type glob -i "$1/*.png"  -c:v libx264 -pix_fmt yuv420p -y -vf scale='floor(iw/2)*2':-2 "$2"
