import nazca as nd
import nazca.geometries as geom
from QRCode import encoderStandard, decoderStandard
from QRCode import encoderMicro, decoderMicro

# ====================================Encoder Exmaple================================
'''
You can use encoder.makeLabelWithQRCode() to create label text with its corresponding Standard QRCode or Micro QRCode
The usage of encoder.makeLabelWithQRCode() is similar to nd.text()
For details of parameters meanings please refer to ./QRCode/encoderStandard.py or ./QRCode/encoderMicro.py
'''
encoderStandard.makeLabelWithQRCode(text='D25S50WG10',height=20,align='cc',QRSize=80,KlayoutDecode=True,layer=2).put(0)
encoderStandard.makeLabelWithQRCode(text='D50S150WG6',height=20,align='cc',QRSize=80,layer=2).put(600)
encoderStandard.makeLabelWithQRCode(text='ABC',height=40,align='cc',layer=2).put(200, 400)
encoderStandard.makeLabelWithQRCode(text='TEST',height=80,align='cc',layer=1).put(-200, 800)

encoderMicro.makeLabelWithQRCode(text='D250S6WG8',height=20,align='cc',layer=3).put(400, 200)

nd.export_gds()


# ====================================Decoder Exmaple================================
'''
You can use decoder.readQRCode() to detect text (label) from Standard QRCode or Micro QRCode in the specified image
Use several different method to detect texts from QRCodes
For details of parameters meanings please refer to ./QRCode/decoderStandard.py or ./QRCode/decoderMicro.py
'''

print('==============================exmaple 1==============================')
print(decoderStandard.readQRCode("./QRCode/example1.png"))
print('==============================exmaple 2==============================')
print(decoderStandard.readQRCode("./QRCode/example2.png"))
print('==============================exmaple 3==============================')
print(decoderStandard.readQRCode("./QRCode/example3.png"))
print('==============================exmaple 4==============================')
print(decoderStandard.readQRCode("./QRCode/example4.png"))

print('==============================exmaple 5==============================')
print(decoderMicro.readQRCode("./QRCode/example6.png"))

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