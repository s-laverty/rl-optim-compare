ffmpeg -i $1 -filter_complex "[0:v] fps=30,scale=480:-1,split [a][b];[a] palettegen [p];[b][p] paletteuse" $1.gif
