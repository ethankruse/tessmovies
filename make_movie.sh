#!/bin/bash
# first argument is directory with input/output files (with trailing /)
# second argument is the name of the output move (without the directory)
# third argument is the frame rate (fps)

# make the mp4 using a particular frame rate.
ffmpeg -framerate $3 -i $1img%05d.png -c:v libx264 -crf 18 $1$2
