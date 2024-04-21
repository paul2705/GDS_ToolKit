import numpy as np
import gdspy

labelStr = input('Please input label (eg.D25S500WG6): ')

with open('./pattern/labels.npy','rb') as file:
    labelList = np.load(file)

gdsFile = gdspy.GdsLibrary(infile='../../Image/Epump-1.4.gds')
# locLine = gdsFile.new_cell('Match Locations')
cell = gdsFile.top_level()[0]

linePolygons = []
for label in labelList:
    if labelStr == label[2]:
        print((label[0], label[1]))
        linePolygons.append(gdspy.Polygon([(float(label[0]), float(label[1])),(float(label[0])+150, float(label[1])),(float(label[0])+150, float(label[1])+10),(float(label[0]), float(label[1])+10)], layer=1))
        # locLine.add(gdspy.Polygon([(float(label[0]), float(label[1])),(float(label[0])+150, float(label[1])),(float(label[0])+150, float(label[1])+10),(float(label[0]), float(label[1])+10)]))

print(gdspy.boolean(linePolygons, None, "xor"))
cell.add(gdspy.boolean(linePolygons, None, "xor"))
# gdspy.LayoutViewer(gdsFile)
gdsFile.write_gds("./result/match.gds")
# writer = gdspy.GdsWriter('./result/match1.gds', unit=1.0e-6, precision=1.0e-9)
# writer.write_cell(cell)
# writer.close()
