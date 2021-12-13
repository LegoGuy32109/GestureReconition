import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions.face_detection import FaceDetection

""" THINGS TO DO:
0. Show prof filming setup, footage of min 4 students both angles
1. track nose, find chin point, more non-deforming parts of face
2. find hand cascade online to track hands
3. if no 2, train a cascade to detect skin color then search for skin color and find hand through another cascade training montage or use cascade that needs skin color to plug in

back of the head for closer students is fine
4. bounding box of the hands to detect when hands merge
5. planar rotation and movement of the head, for group dynamics
    onplane displacement of the head
6. goal towards great showcase video, perfect enviroment, insightful results
"""
# my webcam 
# cap = cv2.VideoCapture(1)

whichVideo = ["Back", "Front"]
video = cv2.VideoCapture(whichVideo[0]+'Angle.mp4')

# gesture resources
mpHands = mp.solutions.hands
# https://google.github.io/mediapipe/solutions/hands.html#max_num_hands yeah doc link baby
hands = mpHands.Hands(static_image_mode = False, max_num_hands = 6)
mpDraw = mp.solutions.drawing_utils

# facial resources
mpFaceDetection = mp.solutions.face_detection
# https://google.github.io/mediapipe/solutions/face_detection.html
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)
faceCascade =  cv2.CascadeClassifier('haar_face.xml')
# eye cascade
eyeCascade = cv2.CascadeClassifier('haar_eye.xml')

# for hand landmark drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
frameId = 0
skippedFrames = 0
while True:
    success, img = video.read()
    if(success):
        if(skippedFrames > 0):
            skippedFrames -= 1
            continue
        #resize it
        scalePercent = 0.5
        frame = cv2.resize(img, (int(img.shape[1]*scalePercent), int(img.shape[0]*scalePercent)))

        #RGBcovert for gestures
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if(results.multi_hand_landmarks):
            for handLms in results.multi_hand_landmarks:
                # draw mediapipe hand annotations to frame
                """
                mp_drawing.draw_landmarks(
                    frame,
                    handLms,
                    mpHands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                """
                # my custom annotations for points on hands
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)

                    # id info:https://medium.com/@sreeananthakannan/hand-tracking-with-21-landmarks-c386beafeaf2#Media%20Pipe
                    # base of hand
                    if id == 0:
                        cv2.circle(frame, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
                    
                    # tip of thumb
                    if id == 4:
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    # tip of index
                    if id == 8:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                #mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        """
        #potentially better facial reconition
        imgRGB.flags.writeable = False
        faceResults = faceDetection.process(imgRGB)
        if faceResults.detections:
            for d in faceResults.detections:
                mpDraw.draw_detection(frame, d)
        """
        #grayscale convert for faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 6,
            minSize = (30, 50)
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,200,0), 2)

            # analyze face for eyes
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            # create eye cascade from this reigon
            eyes = eyeCascade.detectMultiScale(roi_gray, 1.3, 7)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 1)

        # display frame
        cv2.imshow("'q' to quit", frame)
        # cv2.imshow("'q' to quit", imgRGB)
        # cv2.imshow("'q' to quit", gray)
        
        # save frame
        cv2.imwrite('frames2/'+str(frameId)+'.png',frame)
        frameId += 1
    # option to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

