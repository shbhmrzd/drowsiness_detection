# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat


import cv2
import numpy as np
import time
import sys

#blink_detection
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import imutils
import dlib


# blink eye section

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

#sleep time
SLEEP_TIME_TEST = 2

# Flag for testing the start time of the sleep
global isFirstTimeSleep
isFirstTimeSleep = True

global sleepStartTime
sleepStartTime = 0

global isSleeping
isSleeping = False

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]





def blink_detection(frame):
    global TOTAL,COUNTER,isFirstTimeSleep,sleepStartTime,isSleeping

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if(isFirstTimeSleep is True):
                isFirstTimeSleep = False
                sleepStartTime = time.time()

            if ((time.time() - sleepStartTime)>= SLEEP_TIME_TEST):
                isSleeping = True


        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0
            isSleeping = False

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {}".format(ear), (1000, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,"Sleeping = {}".format(isSleeping),(500,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)












# yawn section
path = "classifiers/haar-face.xml"
faceCascade = cv2.CascadeClassifier(path)


# Variable used to hold the ratio of the contour area to the ROI
ratio = 0

# variable used to hold the average time duration of the yawn
global yawnStartTime
yawnStartTime = 0

# Flag for testing the start time of the yawn
global isFirstTime
isFirstTime = True

# List to hold yawn ratio count and timestamp
yawnRatioCount = []

# Yawn Counter
yawnCounter = 0

# yawn time
averageYawnTime = 2

totalYawnCounter = 0

"""
Find the second largest contour in the ROI; 
Largest is the contour of the bottom half of the face.
Second largest is the lips and mouth when yawning.
"""
def calculateContours(image, contours):
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    maxArea = 0
    secondMax = 0
    maxCount = 0
    secondmaxCount = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if maxArea < area:
            secondMax = maxArea
            maxArea = area
            secondmaxCount = maxCount
            maxCount = count
        elif (secondMax < area):
            secondMax = area
            secondmaxCount = count

    return [secondmaxCount, secondMax]

"""
Thresholds the image and converts it to binary
"""
def thresholdContours(mouthRegion, rectArea):
    global ratio

    # Histogram equalize the image after converting the image from one color space to another
    # Here, converted to greyscale
    imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))

    # Thresholding the image => outputs a binary image.
    # Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0.
    ret,thresh = cv2.threshold(imgray, 64, 255, cv2.THRESH_BINARY)

    # Finds contours in a binary image
    # Constructs a tree like structure to hold the contours
    # Contouring is done by having the contoured region made by of small rectangles and storing only the end points
    # of the rectangle
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    returnValue = calculateContours(mouthRegion, contours)

    # returnValue[0] => secondMaxCount
    # returnValue[1] => Area of the contoured region.
    secondMaxCount = returnValue[0]
    contourArea = returnValue[1]

    ratio = contourArea / rectArea

    # Draw contours in the image passed. The contours are stored as vectors in the array.
    # -1 indicates the thickness of the contours. Change if needed.
    if(isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
        cv2.drawContours(mouthRegion, [secondMaxCount], 0, (255,0,0), -1)

"""
Isolates the region of interest and detects if a yawn has occured. 
"""
def yawnDetector(frame):
    global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter,totalYawnCounter


    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Isolate the ROI as the mouth region
        widthOneCorner = (x + (w / 4))
        widthOtherCorner = x + ((3 * w) / 4)
        heightOneCorner = y + (11 * h / 16)
        heightOtherCorner = y + h

        # Indicate the region of interest as the mouth by highlighting it in the window.
        cv2.rectangle(frame, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2)

        # mouth region
        mouthRegion = frame[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]

        # Area of the bottom half of the face rectangle
        rectArea = (w*h)/2

        if(len(mouthRegion) > 0):
            thresholdContours(mouthRegion, rectArea)

        print "Current probablity of yawn: " + str(round(ratio*1000, 2)) + "%"
        print "Length of yawnCounter: " + str(len(yawnRatioCount))

        cv2.putText(frame, "Yawn: {}".format(totalYawnCounter), (1000, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if(ratio > 0.065):
            if(isFirstTime is True):
                isFirstTime = False
                yawnStartTime = time.time()

            # If the mouth is open for more than 2.5 seconds, classify it as a yawn
            if((time.time() - yawnStartTime) >= averageYawnTime):
                yawnCounter += 1
                yawnRatioCount.append(yawnCounter)
        else:
            if(len(yawnRatioCount) > 8):
                # Reset all variables
                totalYawnCounter +=1
                isFirstTime = True
                yawnStartTime = 0
                yawnRatioCount = []
                return True
            isFirstTime = True
            yawnStartTime = 0


    # Display the resulting frame
    # cv2.namedWindow('yawnVideo')
    # cv2.imshow('yawnVideo', frame)
    # time.sleep(0.025)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     sys.exit(0)

    return False


"""
Main
"""
def main():
    # Capture from web camera
    yawnCamera = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = yawnCamera.read()

        returnValueYawn = (yawnDetector(frame), 'yawn')
        if returnValueYawn[0]:
            print "Yawn detected!"
            # cv2.putText(frame, "Yawn Detected", (300, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        blink_detection(frame)
        # Display the resulting frame
        cv2.namedWindow('DrowsinessVideo')
        cv2.imshow('drowsinessVideo', frame)
        time.sleep(0.025)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

        # When everything is done, release the capture
        #yawnCamera.release()
        #cv2.destroyWindow('yawnVideo')
        #return returnValue


main()
