import nazca as nd
import numpy as np
from collections import defaultdict

import GC_SingleFiber
import convert

def makeWaveguideCrossing(layer, taperIn, taperOut, taperLength, insideWaveguideLengthOneSide):
    with nd.Cell(f'Waveguide_Crossing') as aWaveguideCrossing:
        leftMost   = - taperLength - insideWaveguideLengthOneSide - taperOut/2.0
        rightMost  = - leftMost
        bottomMost = leftMost
        upMost     = rightMost

        bottomTaperPoly = [[-taperIn/2.0, leftMost], [-taperOut/2.0, leftMost + taperLength], \
                           [taperOut/2.0, leftMost + taperLength], [taperIn/2.0, leftMost]]
        upTaperPoly     = [[-taperIn/2.0, rightMost], [-taperOut/2.0, rightMost - taperLength], \
                           [taperOut/2.0, rightMost - taperLength], [taperIn/2.0, rightMost]]
        rightTaperPoly = [[upMost - taperLength, -taperOut/2.0], [upMost - taperLength, taperOut/2.0], \
                          [upMost, taperIn/2.0], [upMost, -taperIn/2.0]]
        leftTaperPoly  = [[bottomMost + taperLength, -taperOut/2.0], [bottomMost + taperLength, taperOut/2.0], \
                          [bottomMost, taperIn/2.0], [bottomMost, -taperIn/2.0]]                
        nd.Polygon(layer=layer, points=leftTaperPoly).put()
        nd.Polygon(layer=layer, points=rightTaperPoly).put()
        nd.Polygon(layer=layer, points=upTaperPoly).put()
        nd.Polygon(layer=layer, points=bottomTaperPoly).put()

        # verticalWaveguidePoly   = [[taperOut/2.0, leftMost + taperLength], [taperOut/2.0, rightMost - taperLength], \
        #                            [-taperOut/2.0, rightMost - taperLength], [-taperOut/2.0, leftMost + taperLength]]
        # horizontalWaveguidePoly = [[leftMost + taperLength, taperOut/2.0], [rightMost - taperLength, taperOut/2.0], \
        #                            [rightMost - taperLength, -taperOut/2.0], [leftMost + taperLength, -taperOut/2.0]]
        crossWaveguidePoly = [[leftMost + taperLength, taperOut/2.0], [-taperOut/2.0, taperOut/2.0], [-taperOut/2.0, rightMost - taperLength], \
                              [taperOut/2.0, rightMost - taperLength], [taperOut/2.0, taperOut/2.0], [rightMost - taperLength, taperOut/2.0], \
                              [rightMost - taperLength, -taperOut/2.0], [taperOut/2.0, -taperOut/2.0], [taperOut/2.0, leftMost + taperLength], \
                              [-taperOut/2.0, leftMost + taperLength], [-taperOut/2.0, -taperOut/2.0], [leftMost + taperLength, -taperOut/2.0]]
        nd.Polygon(layer=layer, points=crossWaveguidePoly).put()
        # nd.Polygon(layer=layer, points=verticalWaveguidePoly).put()
    
    return __merge_cell_polygons(aWaveguideCrossing)

def makeWCSeries(WCCell, insertionLoss, totalNumber, rowNumber, gap):
    leftMost     = 0
    leftMostxy   = None
    leftMostpgon = None
    for P in nd.cell_iter(WCCell, flat=True):
        if P.cell_start:
            for pgon, xy, bbox in P.iters['polygon']:
                if leftMost > bbox[0]:
                    leftMost     = bbox[0]
                    leftMostxy   = xy
                    leftMostpgon = pgon
    leftMostxy = np.abs(np.array(leftMostxy))

    grating_coupler = GC_SingleFiber.make_grating_coupler(target_length=15, duty_cycle=0.59036, pitch=0.664, radius=30, y_span=10, L_extra=10, waveguide_width=0.45, waveguide_length=10)
    # grating_coupler.add_port(name = 'output', midpoint = [-10,0], width = 10, orientation = 180)
    GCDesign = convert.convertPhidlToNAZCA(grating_coupler)

    totalLength  = -leftMost * 2.0
    taperInWidth = np.min(leftMostxy[:,1]) * 2.0
    taperOutWidth = np.median(leftMostxy[:,1]) * 2.0
    layer        = leftMostpgon.layer
    connectWaveguidePoly = [[-gap/2.0, -taperInWidth/2.0], [gap/2.0,-taperInWidth/2.0], \
                            [gap/2.0, taperInWidth/2.0], [-gap/2.0, taperInWidth/2.0]]
    
    with nd.Cell(f'Waveguide_Crossing_Series') as WCSeries:
        nd.text(layer=21, text="WC_FP{:.1f}IW{:.1f}OW{:.1f}L{:.3f}C{:d}".format(totalLength,taperInWidth,taperOutWidth,insertionLoss,totalNumber), height=10).put(-5*10,((totalNumber+rowNumber-1)//rowNumber)*(totalLength+1)-5)
        GCDesign.put((totalLength+gap)*(rowNumber-(rowNumber//2)) - (totalLength)/2.0+(5*(2*((totalNumber+rowNumber-1)//rowNumber-1)-1)%np.floor(gap+totalLength)),((totalNumber+rowNumber-1)//rowNumber-1)*(totalLength+1))
        GCDesign.put((-rowNumber//2-1)*(totalLength+gap)-20,0,-180)
        for _ in range((totalNumber+rowNumber-1)//rowNumber):
            tmpRowNum = totalNumber%rowNumber if (_*rowNumber<totalNumber and (_+1)*rowNumber>totalNumber) else rowNumber
            offset    = 5*_%np.floor(gap+totalLength)
            for i in range(-(rowNumber//2), -(rowNumber//2)+tmpRowNum, 1):
                WCCell.put((totalLength+gap)*i+offset,_*(totalLength+1))
                if _==0 and i==-(rowNumber//2):
                    startWaveguidePoly = [[-gap/2.0 - 27.2, -0.45/2.0], [gap/2.0,-taperInWidth/2.0], \
                                            [gap/2.0, taperInWidth/2.0], [-gap/2.0 - 27.2, 0.45/2.0]]
                    nd.Polygon(layer=layer, points=startWaveguidePoly).put((totalLength+gap)*(-(rowNumber//2))-(totalLength+gap)/2.0,0)
                else:
                    nd.Polygon(layer=layer, points=connectWaveguidePoly).put((totalLength+gap)*(i if _%2==0 else i+1) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
            
            if (_*rowNumber<totalNumber and (_+1)*rowNumber>=totalNumber):
                tmpDeltaOffset = (5*(_-1)%np.floor(gap+totalLength)) 
                tmpDeltaOffset = tmpDeltaOffset if _>0 and offset - 5*(_+4)%np.floor(gap+totalLength) >= -gap else tmpDeltaOffset + (gap)//2
                # endWaveguidePoly = [[-gap/2.0, -taperInWidth/2.0], [gap/2.0+(totalLength+gap)*(rowNumber-tmpRowNum)+tmpDeltaOffset,-taperInWidth/2.0], \
                #                     [gap/2.0+(totalLength+gap)*(rowNumber-tmpRowNum)+tmpDeltaOffset, taperInWidth/2.0], [-gap/2.0, taperInWidth/2.0]]
                endWaveguidePoly = [[-gap/2.0, -taperInWidth/2.0], [gap/2.0+(totalLength+gap)*(rowNumber-tmpRowNum)+tmpDeltaOffset,-0.45/2.0], \
                                    [gap/2.0+(totalLength+gap)*(rowNumber-tmpRowNum)+tmpDeltaOffset, 0.45/2.0], [-gap/2.0, taperInWidth/2.0]]
                nd.Polygon(layer=layer, points=endWaveguidePoly).put((totalLength+gap)*(-(rowNumber//2)+tmpRowNum) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
            elif _%2==0:
                tmpDeltaOffset = (5*(_+1)%np.floor(gap+totalLength)) - offset
                tmpDeltaOffset = tmpDeltaOffset if tmpDeltaOffset>=-gap else tmpDeltaOffset + (gap+totalLength)
                endWaveguidePoly = [[-gap/2.0, -taperInWidth/2.0], [gap/2.0+tmpDeltaOffset,-taperInWidth/2.0], \
                                    [gap/2.0+tmpDeltaOffset, taperInWidth/2.0], [-gap/2.0, taperInWidth/2.0]]
                nd.Polygon(layer=layer, points=endWaveguidePoly).put((totalLength+gap)*((tmpRowNum+1)//2) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
                if _>0 and offset - 5*(_+4)%np.floor(gap+totalLength) > gap:
                    tmpWaveguidePoly = [[-gap/2.0 - (gap+totalLength), -taperInWidth/2.0], [gap/2.0,-taperInWidth/2.0], \
                                        [gap/2.0, taperInWidth/2.0], [-gap/2.0 - (gap+totalLength), taperInWidth/2.0]]
                    nd.Polygon(layer=layer, points=tmpWaveguidePoly).put((totalLength+gap)*(-(tmpRowNum)//2) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
                nd.bend(layer=layer, width=taperInWidth, angle=180, radius=(totalLength+1)/2.0).put((totalLength+gap)*((tmpRowNum+1)//2) - (totalLength)/2.0+offset+tmpDeltaOffset,_*(totalLength+1))
            else:
                tmpDeltaOffset = (5*(_+1)%np.floor(gap+totalLength)) - offset
                tmpDeltaOffset = tmpDeltaOffset if tmpDeltaOffset<=gap else tmpDeltaOffset - (gap+totalLength)
                endWaveguidePoly = [[-gap/2.0 + tmpDeltaOffset, -taperInWidth/2.0], [gap/2.0,-taperInWidth/2.0], \
                                    [gap/2.0, taperInWidth/2.0], [-gap/2.0 + tmpDeltaOffset, taperInWidth/2.0]]
                nd.Polygon(layer=layer, points=endWaveguidePoly).put((totalLength+gap)*(-(tmpRowNum)//2) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
                if _>0 and offset - 5*(_+4)%np.floor(gap+totalLength) < -gap:
                    tmpWaveguidePoly = [[-gap/2.0, -taperInWidth/2.0], [gap/2.0 + (gap+totalLength),-taperInWidth/2.0], \
                                        [gap/2.0 + (gap+totalLength), taperInWidth/2.0], [-gap/2.0, taperInWidth/2.0]]
                    nd.Polygon(layer=layer, points=tmpWaveguidePoly).put((totalLength+gap)*((tmpRowNum+1)//2) - (totalLength+gap)/2.0+offset,_*(totalLength+1))
                nd.bend(layer=layer, width=taperInWidth, angle=-180, radius=(totalLength+1)/2.0).put((totalLength+gap)*(-(tmpRowNum)//2) - (totalLength)/2.0 - gap+offset + tmpDeltaOffset,_*(totalLength+1),-180)
    return __merge_cell_polygons(WCSeries)

def __merge_cell_polygons(cell):
    """Flatten a NAZCA cell and merge polygons per layer.
   
    Args:
        cell (Cell): NAZCA cell to flatten and merge polygon of.

    Returns:
        Cell: NAZCA flattened cell with merged polygons per layer.
    """
    layerpgons = defaultdict(list)
    for P in nd.cell_iter(cell, flat=True):
        if P.cell_start:
            for pgon, xy, bbox in P.iters['polygon']:
                layerpgons[pgon.layer].append(xy)
    with nd.Cell(name=f"{cell.cell_name}_merged") as C:
        for layer, pgons in layerpgons.items():
            merged = nd.clipper.merge_polygons(pgons)
            for pgon in merged:
                nd.Polygon(points=pgon, layer=layer).put(0)
    return C
    
def makeWaveguideCrossingDesign(posX, posY):
    
    WCCell   = makeWaveguideCrossing(layer=3, taperIn=0.7, taperOut=1.9, taperLength=2.3, insideWaveguideLengthOneSide=(9.7-1.9)/2.0)
    WCSeries = makeWCSeries(WCCell, insertionLoss=0.048, totalNumber=63, rowNumber=10, gap=10)
    WCSeries.put(posX,posY)
    WCSeries.put(posX,posY-150)

    WCSeries = makeWCSeries(WCCell, insertionLoss=0.048, totalNumber=105, rowNumber=16, gap=10)
    WCSeries.put(posX+500,posY)
    WCSeries.put(posX+500,posY-150)

if __name__ == '__main__':

    makeWaveguideCrossingDesign(posX=0, posY=0)
    nd.export_gds()