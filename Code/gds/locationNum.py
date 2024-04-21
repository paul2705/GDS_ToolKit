import gdspy
import numpy as np
import os

gdsFile = gdspy.GdsLibrary(infile='../../Image/Epump-1.4.gds')
polygons = gdsFile.top_level()[0].get_polygons()

cnt = 0
shapeDict = dict()
for r, d, f in os.walk('./pattern'):
    for file in f:
        if file.endswith(".gds"):
            print(file,os.path.join(r, file))
            tmpGds = gdspy.GdsLibrary(infile=os.path.join(r, file))
            shapeDict[file[0]] = tmpGds.top_level()[0].get_polygons()[0]
            xmin = shapeDict[file[0]][np.argmin(tmpGds.top_level()[0].get_polygons()[0],axis=0)[0],0]
            xmax = shapeDict[file[0]][np.argmax(tmpGds.top_level()[0].get_polygons()[0],axis=0)[0],0]
            ymin = shapeDict[file[0]][np.argmin(tmpGds.top_level()[0].get_polygons()[0],axis=1)[0],1]
            ymax = shapeDict[file[0]][np.argmax(tmpGds.top_level()[0].get_polygons()[0],axis=1)[0],1]
            print(xmin,ymin,xmax,ymax)
            # newGds = gdspy.GdsLibrary()
            # cell = newGds.new_cell('{:d}'.format(cnt))
            # cell.add(gdspy.boolean([shapeDict[file[0]]], None, "xor"))
            # gdspy.LayoutViewer(newGds)
            # cnt = cnt + 1

labelDict = dict()
for shape in polygons:
    vecShape = shape - shape[0]
    foundShape = None
    for sample in shapeDict:
        if vecShape.shape == shapeDict[sample].shape and np.sqrt(np.sum(np.sum((vecShape - shapeDict[sample])**2))) < 45:
            foundShape = sample
            break
    if foundShape is None:
        continue
    foundLabel = None
    for label in labelDict:
        if np.sqrt(np.sum((shape[0] - label)**2)) < 150:
            labelDict[label].append((foundShape,shape[0][0]))
            foundLabel = label
            break
    if foundLabel is not None:
        continue
    labelDict[(shape[0][0],shape[0][1])] = [(foundShape,shape[0][0])]
    
labels = []
cnt = 0
for label in labelDict:
    labelDict[label].sort(key=lambda tuple : tuple[1])
    labelPosition = (labelDict[label][0][1], label[1])
    labelStr = ''
    for figure in labelDict[label]:
        labelStr = labelStr + figure[0]
    labels.append([labelPosition[0], labelPosition[1], labelStr])
    cnt = cnt + 1

with open('./pattern/labels.npy','wb') as file:
    np.save(file, np.array(labels))

print('Detected Labels has been saved to labels.npy, save format [...,(labelPosition_x,labelPosition_y,label_text),...]')

# with open('labels.npy','rb') as file:
#     a = np.load(file)
# print(a)