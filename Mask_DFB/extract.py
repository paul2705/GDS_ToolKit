import gdspy
import numpy as np
import nazca as nd

gdsFile = gdspy.GdsLibrary(infile='./DFB1StepperL2Dark.gds')
polygons = gdsFile.top_level()[0].get_polygons()

cnt = 0
shapeDict = dict()
for shape in polygons:
    if len(shape) == 7988:
        shape[:,0]-=np.mean(shape[:,0])
        # # shape[:,0]/=24.0
        shape[:,1]-=np.mean(shape[:,1])
        maxx = np.max(shape[:,0])
        miny = np.min(shape[:,1])
        maxy = np.max(shape[:,1])
        # print(minx, miny, maxy)
        # # shape[:,1]/=24.0
        newshape = nd.clipper.polygons_AND([list(shape)],[[(maxx,miny),(maxx,maxy),(488,maxy),(488,miny)]])
        # print(newshape)
        # nd.Polygon(layer=2, points=list(map(tuple, shape))).put()
        for _ in range(len(newshape)):
            # if (len(newshape[_])>2000):
            #     continue

            tmpshape = np.array(newshape[_])
            tmpshape[:,0]-=np.mean(tmpshape[:,0])
            tmpshape[:,1]-=np.mean(tmpshape[:,1])
            tmpshape/=6
            nd.Polygon(layer=2, points=list(map(tuple, tmpshape))).put()
        nd.export_gds()
        exit(0)
    # shape = shape - shape[0]
    # found = 0
    # for sample in shapeDict:
    #     if shape.shape == shapeDict[sample].shape and np.sqrt(np.sum(np.sum((shape - shapeDict[sample])**2))) < 45:
    #         found = 1
    #         break
    # if found == 1:
    #     continue
    # cnt = cnt + 1
    # shapeDict[cnt] = shape