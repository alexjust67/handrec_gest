import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import pickle
import time

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_EXPOSURE, -6)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def handlandmarker(image):
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image)#cv.cvtColor(image, cv.COLOR_RGB2GRAY)


    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv.imshow("tracked",annotated_image)
    
    return(detection_result)
    

def contours(frame):
    
    
    ret, thresh = cv.threshold(frame, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[5]
    
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    #cv.drawContours(frame, [cnt], 0, (0,255,0), 3)
normal=[]
ite=0
while ite<80:
    
    
    gesture='openplm_test'
    normal.append([[],[]])
    
    while True:
        try:
            ret, framen = cap.read()
            i=handlandmarker(framen)
            
            for a in i.hand_landmarks[0]:
                
                normal[ite][1].append(a.x)
                normal[ite][1].append(a.y)
                normal[ite][1].append(a.z)
            normal[ite][0].append('openplm')
            print(normal)
            break
        except Exception as e:
        
            print(f"An error occurred: {e}")

    time.sleep(1)
        
    print(ite)
    ite+=1
    c = cv.waitKey(1)
    if c == 27:
        break

with open(gesture, "wb") as fp:   #Pickling
    pickle.dump(normal, fp)

cap.release()
cv.destroyAllWindows()

