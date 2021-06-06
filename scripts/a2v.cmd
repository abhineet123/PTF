ffmpeg -loop 1 -i "C:\Users\Tommy\Pictures\Colors\black.png" -i %1 -c:a copy -c:v libx264 -shortest out.mp4
