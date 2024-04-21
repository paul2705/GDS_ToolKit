from klayout import lay, db
import numpy as np
import os

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

# Save the image
TEXT_WIDTH = 12.6
DATA_NUM   = 1e3

with open('./pattern/labels.npy','rb') as file:
    labelList = np.load(file)

cnt = 0
for labelTuple in labelList:
    T = np.random.randint(len(labelTuple[2]), size=1)[0] + 1
    while T!=0:
        T -= 1
        # startNum = np.random.randint(len(labelTuple[2]), size=1)[0]
        # displayNum = np.random.randint(len(labelTuple[2])-startNum, size=1)[0] + 1
        startNum = 0
        displayNum = len(labelTuple[2])
        startXPos = float(labelTuple[0])
        startYPos = float(labelTuple[1])
        res = np.random.randint(40, size=1)[0]
        bbox = db.DBox(db.DPoint(startXPos+startNum*TEXT_WIDTH, startYPos+20+res), db.DPoint(startXPos+(startNum+displayNum)*TEXT_WIDTH, startYPos))
        w = 400
        h = int(0.5 + w * bbox.height() / bbox.width() +res )
        lineWidth = int(np.random.randint(10, size=1)[0])
        lv.save_image_with_options(os.path.join(outPath,"test{:04d}.png".format(cnt)), w, h, lineWidth, 0, 0, bbox, False)
        with open(os.path.join(outPath,"test{:04d}.gt.txt".format(cnt)),'w') as textFile:
            textFile.write(labelTuple[2][startNum:startNum+displayNum])  
        print("test{:04d}".format(cnt))
        cnt += 1
        if cnt >= DATA_NUM:
            break
    if cnt >= DATA_NUM:
        break
    
# lv.save_image(out_file, 600, 400)
# x = 1358.346
# y = 1737.246
# bbox = db.DBox(db.DPoint(x, y+20), db.DPoint(x+12.6*10, y))
# w = 400
# h = int(0.5 + w * bbox.height() / bbox.width())
# lv.save_image_with_options(os.path.join(outPath,"out3.png"), w, h, 0, 0, 0, bbox, False)


# x = 2542.836
# y = -3067.752
# bbox = db.DBox(db.DPoint(x, y+20), db.DPoint(x+12.6*10, y))
# w = 400
# h = int(0.5 + w * bbox.height() / bbox.width())
# lv.save_image_with_options(os.path.join(outPath,"out4.png"), w, h, 0, 0, 0, bbox, False)