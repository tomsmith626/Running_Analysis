import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# User inputs the name of the video they want to analyse
# Need an error catch if an incorrect entry made by user
video_name = input("Please type the filename of the video you would like to use, including its filetype: ")
video = ("Videos/" + video_name)
cap = cv.VideoCapture(video)

# Check if camera is open
if (cap.isOpened()== False):
    print("Camera not open")
    
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Read the entire file until it is completed 
    while(cap.isOpened()): 
    # Capture each frame 
        ret, frame = cap.read() 
        if ret == True: 
            # To improve performance, optionally mark the frame as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = holistic.process(frame)

            # Draw landmark annotation on the frame.
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            #mp_drawing.draw_landmarks(
            #    frame,
            #    results.face_landmarks,
            #    mp_holistic.FACEMESH_CONTOURS,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles
            #    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            # Flip the frame horizontally for a selfie-view display.
            cv.imshow('Running with markers. Press \'Q\' to exit.', frame)
            
        # Press Q on keyboard to exit 
            if cv.waitKey(25) & 0xFF == ord('q'): 
                break
    
    # Break the loop 
        else: 
            break
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv.destroyAllWindows() 