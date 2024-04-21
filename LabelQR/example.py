import nazca as nd
import nazca.geometries as geom
from QRCode import encoder, decoder

# ====================================Encoder Exmaple================================
'''
You can use encoder.makeLabelWithQRCode() to create label text with its corresponding QRCode
The usage of encoder.makeLabelWithQRCode() is similar to nd.text()
For details of parameters meanings please refer to ./QRCode/encoder.py
'''
encoder.makeLabelWithQRCode(text='D25S50WG10',height=20,align='cc',QRSize=80,layer=2).put(0)
encoder.makeLabelWithQRCode(text='ABC',height=40,align='cc',layer=2).put(200, 400)
encoder.makeLabelWithQRCode(text='TEST',height=80,align='cc',layer=1).put(-200, 800)
nd.export_gds()


# ====================================Decoder Exmaple================================
'''
You can use decoder.readQRCode() to detect text (label) from QRCode in the specified image
Use several different method to detect texts from QRCodes
For details of parameters meanings please refer to ./QRCode/decoder.py
'''
print('==============================exmaple 1==============================')
print(decoder.readQRCode("./QRCode/example1.png"))
print('==============================exmaple 2==============================')
print(decoder.readQRCode("./QRCode/example2.png"))
print('==============================exmaple 3==============================')
print(decoder.readQRCode("./QRCode/example3.png"))
print('==============================exmaple 4==============================')
print(decoder.readQRCode("./QRCode/example4.png"))

# ====================================nazca exmaple================================
# frame = nd.Polygon(layer=19, points=[(0,0), (220,0), (220,180), (0,180)])
# boat = nd.Polygon(layer=56, points=[(0,0), (120,0), (130,40), (-10, 40)])
# sail = nd.Polygon(layer=52, points=[(0,0), (100,0), (90,100)])

# sun = nd.Polygon(layer=36, points=geom.circle(radius=25))
# hole = nd.Polygon(layer=60, points=geom.ring(radius=5))

# # position the polygons
# frame.put(-10, 0, 0)
# boat.put(40, 20, 0)
# sail.put(50, 70, -10)
# sun.put(40,110)
# hole.put(10,160,0)
# hole.put(190,160,0)


# message = "Nazca, open source Photonic IC design in Python 3!"
# for i in range(7):
#     T1 = nd.text(text=message, height=70*0.85**i, align='lb', layer=1)
#     T1.put(0, -i*100)
#     T1.put(0, -1200+i*100)

#     T2 = nd.text(text=message, height=70*0.85**i, align='rb', layer=1)
#     T2.put(4000, -i*100)
#     T2.put(4000, -1200+i*100)