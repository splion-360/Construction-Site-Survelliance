import cv2 as cv
import sys
import numpy as np
import os.path
import os
import time
import matplotlib.pyplot as plt
import statistics
label = ""
class BoundingBox:

    def getOutputsNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def get_class_names(self, fname):
        classesFile = fname
        with open(classesFile, mode='r+') as file:
          classes = file.readlines()
        class_names = [c.strip() for c in classes]
        return class_names

    def helmet_color_boxes_avg(self, left, top, right, bottom, img):
        r_list = []
        g_list = []
        b_list = []
        for x_coord in range(img.shape[0]):
            for y_coord in range(img.shape[1]):
                r_list.append(img[x_coord][y_coord][0])
                g_list.append(img[x_coord][y_coord][1])
                b_list.append(img[x_coord][y_coord][2])
        r_mean = int(statistics.mode(r_list))
        b_mean = int(statistics.mode(b_list))
        g_mean = int(statistics.mode(g_list))

        cv.rectangle(img, (left, top), (right, bottom), (r_mean, g_mean, b_mean), 3)
        return b_mean, g_mean, r_mean

    def net_define(self, model_cfg, model_wgts):
      net = cv.dnn.readNetFromDarknet(model_cfg, model_wgts)
      net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
      net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
      return model_cfg, model_wgts, net

    def drawPred(self, classId, conf, classes, left, top, right, bottom, frame, color=False):
        global label
        if color:
          b,g,r, = self.helmet_color_boxes_avg(left, top, right, bottom, frame)
        else:
          cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        if color:
          cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (r, g, b), cv.FILLED)
        else:
           cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)



    def postprocess(self, frame, outs, confThreshold, nmsThreshold, classes):
        global label
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        detection_count = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if detection[4] > confThreshold:
                    detection_count += 1

                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(classIds[i], confidences[i], classes, left, top, left + width, top + height, frame, color=True)

        return label



