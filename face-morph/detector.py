# comes from http://dlib.net/face_landmark_detection.py.html
import sys
import os
import dlib

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("./dlib-detector-demo/shape_predictor_68_face_landmarks.dat")

def getPoints(image):
	dets=detector(image,1)
	data,=dets
	path=predictor(image,data)
	return [(p.x,p.y) for p in path.parts()]