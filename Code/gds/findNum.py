import gdspy
import numpy as np

gdsFile = gdspy.GdsLibrary(infile='../../Image/Epump-1.4.gds')
polygons = gdsFile.top_level()[0].get_polygons()

cnt = 0
shapeDict = dict()
for shape in polygons:
    shape = shape - shape[0]
    found = 0
    for sample in shapeDict:
        if shape.shape == shapeDict[sample].shape and np.sqrt(np.sum(np.sum((shape - shapeDict[sample])**2))) < 45:
            found = 1
            break
    if found == 1:
        continue
    cnt = cnt + 1
    shapeDict[cnt] = shape

print(cnt)

cnt = 0
for shape in shapeDict:
    newGds = gdspy.GdsLibrary()
    cell = newGds.new_cell('{:d}'.format(cnt))
    cell.add(gdspy.boolean([shapeDict[shape]], None, "xor"))
    gdspy.LayoutViewer(newGds)
    responseStr = input("Identify if this is a text (if not input '/'):")
    if responseStr != '/' and responseStr != '':
        newGds.write_gds("./pattern/{:s}.gds".format(responseStr))
    newGds.remove(cell)
    cnt = cnt + 1