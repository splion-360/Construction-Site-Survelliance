from djitellopy import tello
import time
import cv2
me = tello.Tello()
me.connect()
me.streamon()
def detectface(cascade,img):
    face_img = img.copy()
    rect = cascade.detectMultiScale(face_img)
    for (x,y,w,h) in rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),color=(0,0,255),thickness = 5)
    print(rect)
    return face_img
while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img,(512,512))
    face = cv2.CascadeClassifier('C:\\Users\\SuryaPrakash\\Desktop\\New folder\\Computer-Vision-with-Python\\DATA\\haarcascades\\haarcascade_frontalface_default.xml')
    img = detectface(face,img)

    cv2.imshow('Tello',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()