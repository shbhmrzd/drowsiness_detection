# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat


#for pushing messages to kafka
from pykafka import KafkaClient

import cv2
import numpy as np
import time
import sys
import datetime
#blink_detection
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import imutils
import dlib
import pickle
import os





# for kafka pushing messages
msgs = []


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


#arguments for face detection and identification
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")


#arguments for blink detection
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")

#parse the args
args = vars(ap.parse_args())




#face detection and identification start

# load our serialized face detector from disk

global protoPath,modelPath,detector,embedder,recognizer

#face detection and identification end









# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

#sleep time
SLEEP_TIME_TEST = 5.0

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
blink_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]





def blink_detection(frame):
    global TOTAL,COUNTER,isFirstTimeSleep,sleepStartTime,isSleeping

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = blink_detector(gray, 0)

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

            elif ((time.time() - sleepStartTime)>= SLEEP_TIME_TEST):
                isSleeping = True
                isFirstTimeSleep = True


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
        cv2.putText(frame, "Eye Ratio: {:.2f}".format(ear), (1000, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame,"Sleeping = {}".format(isSleeping),(10,200),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)












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

yawnThreshold = 0.06

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
    global ratio, yawnStartTime, isFirstTime, yawnRatioCount, yawnCounter,totalYawnCounter,yawnThreshold


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
        #was creating a green rectangle, we dont need it as the face detecter creates one now
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

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

        cv2.putText(frame, "Mouth Ratio: {:.2f}".format(ratio), (1000, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if(ratio > yawnThreshold):
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







def publish_to_kafka():
    global TOTAL,totalYawnCounter,isSleeping,msgs
    message = {}
    message['event_ts'] = str(datetime.datetime.now())
    message['event_type'] = 'inside_camera'
    payload = {}
    payload['blink_count'] = TOTAL
    payload['yawn_count'] = totalYawnCounter
    payload['is_sleeping'] = isSleeping
    message['payload'] = payload
    message['trip_id'] = '123'
    message['driver_id'] = 'D#123'
    msgs.append(str(message))


"""
Main
"""
def main():




    # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    # time.sleep(2.0)
    # Capture from web camera
    global protoPath,modelPath,detector,embedder,recognizer
    yawnCamera = cv2.VideoCapture(0)
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])

    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())


    while True:
        # Capture frame-by-frame
        ret, frame = yawnCamera.read()



        """face identification"""


        # frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)











        """yawn detection"""

        returnValueYawn = (yawnDetector(frame), 'yawn')
        if returnValueYawn[0]:
            print "Yawn detected!"
            # cv2.putText(frame, "Yawn Detected", (300, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        """blink detection"""

        blink_detection(frame)
        # Display the resulting frame
        cv2.namedWindow('DriverVideo')
        cv2.imshow('driverVideo', frame)
        publish_to_kafka()
        time.sleep(0.025)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        """persisting the message to send later"""


        # When everything is done, release the capture
        #yawnCamera.release()
        #cv2.destroyWindow('yawnVideo')
        #return returnValue

    #outside while loop
    """Sending messages to kafka"""
    kafka_client = KafkaClient(hosts="dfw-kafka-broker00-cp.staging.walmart.com:9092")
    kafka_topic = kafka_client.topics['simsds_items_updated_dev']
    kafka_producer = kafka_topic.get_sync_producer()
    print("length kafka message")
    print(len(msgs))
    for msg in msgs:
        kafka_producer.produce(msg)


main()
