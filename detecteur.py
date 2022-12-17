# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 00:20:07 2020

@author: T.i.Cissé
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
from collections import OrderedDict


def sound_alarm(path):
	playsound.playsound(path)

def cal_E_A_R(eye):
	dist_a = dist.euclidean(eye[1], eye[5])
	dist_b = dist.euclidean(eye[2], eye[4])
	dist_c = dist.euclidean(eye[0], eye[3])

	e_a_r = (dist_a + dist_b) / (2.0 * dist_c)
	return e_a_r


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--marqueur", 
	type=str,default="contour.dat",
	help="chemin vers le fichier délimiteur de contour du visage")
ap.add_argument("-a", "--alarme", 
	type=str, default="sonnerie.mp3",
	help="chemin vers le fichier alarme")
ap.add_argument("-w", "--webcam", 
	type=int, default=0,
	help="index de la webcam utilisée")

args = vars(ap.parse_args())

print("marqueur ",args["marqueur"])
print("alarme ",args["alarme"])
print("webcam ", args["webcam"]+1)


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False


print("[INFO] chargement du fichier délimiteur de contour...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["marqueur"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] Ouverture de la fenêtre de video stream...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)	
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	overlay = image.copy()
	output = image.copy()
    
	if colors is None:
		colors = [(19, 199, 109), 
			(79, 76, 240), 
			(230, 159, 23),
			(168, 100, 168), 
			(158, 163, 32),
			(163, 38, 32), 
			(180, 42, 220)]

	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]

		if name == "jaw":
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	return output


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)
	cv2.putText(frame, "Press 'q' to exit", (600, 550),
					cv2.FONT_HERSHEY_SIMPLEX, 
					0.7, (0, 255, 255), 2)
    
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			
			for (x, y) in shape[i:j]:
				cv2.circle(frame, (x, y),
				1, (0, 0, 255), -1)
                
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = frame[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=600, 
				inter=cv2.INTER_CUBIC)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		left_EAR = cal_E_A_R(leftEye)
		right_EAR = cal_E_A_R(rightEye)

		e_a_r = (left_EAR + right_EAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, 
			(0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, 
			(0, 255, 0), 1)
	
		if e_a_r < EYE_AR_THRESH:
			COUNTER += 1
			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				
				if not ALARM_ON:
					ALARM_ON = True

					if args["alarme"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarme"],))
						t.deamon = True
						t.start()

				cv2.putText(frame, "ATTENTION, VOUS DORMEZ", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7,
					(0, 0, 255), 2)
					
		else:
			COUNTER = 0
			ALARM_ON = False
			cv2.putText(frame, "VOUS ETES EVEILLE", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 255, 0), 2)

		cv2.putText(frame, "".format(e_a_r), 
				(400, 30),
				cv2.FONT_HERSHEY_TRIPLEX, 
				0.7, (0, 255, 0), 2)
	

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()