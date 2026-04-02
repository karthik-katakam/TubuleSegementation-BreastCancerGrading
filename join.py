import pyvips
import os

tiles_across = 171 
tiles_down = 155 

tiles = [pyvips.Image.new_from_file(f"./17_mask_png/{x}_{y}.png", access="sequential")
         for y in range(tiles_down) for x in range(tiles_across)]
im = pyvips.Image.arrayjoin(tiles, across=tiles_across)

im.write_to_file("t96_mask.tiff")





