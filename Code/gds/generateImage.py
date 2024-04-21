from klayout import lay, db
import numpy as np
import os

# import sys
# sys.path.append('../')
# from QRCode import encoder, decoder
# sys.path.append('./gds')

inFile = "../../Image/Epump-1.4.gds"
outPath = "./gds-ground-truth/"
# layer_file = "path/to/some/layers.lyp"

# Set display configuration options
lv = lay.LayoutView()
lv.set_config("background-color", "#ffffff")
lv.set_config("grid-visible", "false")
lv.set_config("grid-show-ruler", "false")
lv.set_config("text-visible", "false")


# Load the GDS and layer files
lv.load_layout(inFile, 0)

# lv.load_layer_props(layer_file)
lv.max_hier()
# Important: event processing for delayed configuration events
# Here: triggers refresh of the image properties
lv.timer()

x = 2400.836
y = -3067.752
bbox = db.DBox(db.DPoint(x, y+300), db.DPoint(x+400, y))
w = 400
h = int(0.5 + w * bbox.height() / bbox.width())
lv.save_image_with_options(os.path.join(outPath,"out4.png"), w, h, 0, 0, 0, bbox, False)