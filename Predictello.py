from BoundingBox import *
import cv2 as cv
from djitellopy import tello
import serial
import KPM as kp
from Test import *

kp.init()
me = tello.Tello()
bb = BoundingBox()

me.connect()
me.streamon()
me.takeoff()

confThreshold = 0.15
nmsThreshold = 0.5

inpWidth = 608
inpHeight = 608

classes = bb.get_class_names('C:\\Users\\SuryaPrakash\\Desktop\\New folder\\darknet\\obj.names')
modelConfiguration, modelWeights, net = bb.net_define('C:\\Users\\SuryaPrakash\\Desktop\\New folder\\darknet\\cfg\\yolov3.cfg', 'C:\\Users\\SuryaPrakash\\Desktop\\New folder\\darknet\\backup\yolov3_last.weights')

database = {}
model = load_model('C:\\Users\\SuryaPrakash\\Downloads\\Facemodel.h5')
model.load_weights('C:\\Users\\SuryaPrakash\\Downloads\\Weights.h5')
database['Surya'] = img_to_encoding('C:\\Users\\SuryaPrakash\\Desktop\\Drone project\\WhatsApp Image 2021-10-02 at 10.44.46 AM (1).jpeg', model)
database['Srihari'] = img_to_encoding('C:\\Users\\SuryaPrakash\\Downloads\\WhatsApp Image 2021-10-03 at 11.00.04 AM.jpeg', model)

def getkeyboard():
    lr, fb, ud, yaw = 0, 0, 0, 0
    speed = 20
    if kp.getkey("LEFT"):
        lr = -speed
    elif kp.getkey("RIGHT"):
        lr = speed
    if kp.getkey("UP"):
        ud = speed
    elif kp.getkey("DOWN"):
        ud = -speed
    if kp.getkey("w"):
        fb = speed
    elif kp.getkey("s"):
        fb = -speed
    if kp.getkey("a"):
        yaw = -speed
    if kp.getkey("d"):
        yaw = speed
    return [lr, fb, ud, yaw]

arduino = serial.Serial(port='COM3', baudrate=115200, timeout=.1)

def write(byteid):
    arduino.write(bytes(byteid, 'utf-8'))
    time.sleep(0.05)

while True:
    vals = getkeyboard()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    img = me.get_frame_read().frame
    blob = cv.dnn.blobFromImage(img, 1 / 255, (inpHeight, inpWidth), [0, 0, 0], swapRB=1, crop=False)
    net.setInput(blob)
    outs = net.forward(bb.getOutputsNames(net))
    label = bb.postprocess(img, outs, confThreshold, nmsThreshold, classes)

    if label[:4] == 'head':
        write('1')

    #    dist_s = verify(img, "Surya", database, model)
    ##    dist_r = verify(img, "Srihari", database, model)
    #    if dist_s > 0.5 and dist_r < 0.5:

    #    elif dist_s <0.5 and dist_r>0.5:
    #        write('2')
    #    elif dist_s < 0.5 and dist_r < 0.5:
    #        write('3')
    #    else:
    #        print("Please go away")

    if cv.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
    cv.imshow('Tello', img)

cv.destroyAllWindows()



