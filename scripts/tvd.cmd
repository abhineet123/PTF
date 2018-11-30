for %f in (*.%1) do @ffmpeg -i %f 2>&1 | findstr Duration > result.txt
for %f in (*.mkv) do @ffmpeg -i %f 2>&1 | findstr Duration > result.txt
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 coyote_doc_1.mkv