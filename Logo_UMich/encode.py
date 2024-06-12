from collections import defaultdict
import nazca as nd
import nazca.geometries as geom
from PIL import Image
import pathlib
import numpy as np
from matplotlib import pyplot as plt

def make(height, align, Size=None, layer=20, KlayoutDecode=False, QRSavePath=None):
    """An extended function to render a QRCode with given text
   
    Args:
        text (String): the text that QRCode image contains
        height (Float): height of the text in um, examples: 20
        align (String): relative placement, examples: 'lc', 'cc', 'tr'
        (Optional) QRSize (Float): QRCode size will be (QRSize,QRSize) in um, default will be (4*height, 4*height)
        (Optional) layer (Int): layer reference to generate label and QRCode, default will be 20
        (Optional) KlayoutDecode: If it is Klayout View (KlayoutDecode=1), we implement some method to detect QRCode. Default: False
        (Optional) QRSavePath (String): the path you want to save created QRCode (If you want to save it)

    Returns:
        Cell: flattened NAZCA cell with text ploygons and QRCode ploygons.
    """
    image = np.array(Image.open('./umich.png').convert('RGB'))
    N, M, C = image.shape
    # colList = Image.Image.getcolors(Image.open('./umich.png').convert('RGB'),maxcolors=1024)
    # dtype = [('count', int), ('rbg', tuple)]
    # newList = np.array(colList, dtype=dtype)
    # sortList = np.sort(newList, order='count')
    # print(sortList)
    # plt.plot(image)
    # plt.show()

    # image.convert('RGB')
    
    # ly = db.Layout()
    # ly.dbu = 0.001
    # top_cell = ly.create_cell("TOP")
    # image = lay.Image("test_QR_8.png")

    # scalor for converting pixel to layout unit (um)
    if Size is None:
        Size = 4*height
    pixelSize = Size / np.max([N, M])

    # image_geo = db.Region()
    with nd.Cell(f'logo_UMich') as logo:
        startPoint = (0, 0)
        for y in range(0, N):
            on = False
            xstart = 0
            for x in range(0, M):
                # value = (image[y, x, :] == [248, 230, 24]).any() #yellow
                value = (image[y, x, :] == [18, 24, 73]).any() #blue
                # print(value)
                if value != on:
                    on = value
                    if on: 
                        xstart = x + 3*KlayoutDecode
                    else:
                        # image_geo.insert(db.Box(xstart, y, int(np.max([x-3,xstart])), y + 1) * pixelSize)
                        xl = xstart * pixelSize
                        xr = int(np.max([x - 3*KlayoutDecode, xstart])) * pixelSize
                        yl = -y * pixelSize
                        yr = (-y - 1) * pixelSize
                        if not ((yl==35750) and (yr==35781)):
                            nd.Polygon(layer=layer, points=[(xl, yl), (xr, yl), (xr, yr), (xl, yr)]).put(startPoint)
            # EDIT: added these two lines
            if on: 
                # image_geo.insert(db.Box(xstart, y, image.width(), y + 1) * pixelSize)
                xl = xstart * pixelSize
                xr = int(np.max([M - 3*KlayoutDecode, xstart])) * pixelSize
                yl = -y * pixelSize
                yr = (-y - 1) * pixelSize
                if not ((yl==35750) and (yr==35781)):
                    nd.Polygon(layer=layer, points=[(xl, yl), (xr, yl), (xr, yr), (xl, yr)]).put(startPoint)

    # image_geo = image_geo.merged()
    # layer = ly.layer(1, 0)
    # top_cell.shapes(layer).insert(image_geo)

    # image_geo = image_geo.smoothed(pixel_size * 0.99)
    # layer = ly.layer(1, 0)
    # top_cell.shapes(layer).insert(image_geo)
    # ly.write("converted.gds")
    return __merge_cell_polygons(logo)

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

if __name__=='__main__':
    logo = make(height=40,align='cc',layer=0)

    with nd.Cell(name=f"logoImproved") as C: #blue
        for P in nd.cell_iter(logo, flat=True):
            if P.cell_start:
                for pgon, xy, bbox in P.iters['polygon']:
                    if (np.array(xy).shape[0])>5:
                        nd.Polygon(points=xy, layer=0).put(0)
    C.put()

    nd.export_gds()