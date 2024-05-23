import cv2 as cv
import numpy as np
import mediapipe as mp
import napari
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

xrh = []
yrh = []
zrh = []
rh = []

xrk = np.array([])
yrk = np.array([])
zrk = np.array([])

xlh = np.array([])
ylh = np.array([])
zlh = np.array([])

xlk = np.array([])
ylk = np.array([])
zlk = np.array([])

xra = np.array([])
yra = np.array([])
zra = np.array([])

xla = np.array([])
yla = np.array([])
zla = np.array([])

xls = np.array([])
yls = np.array([])
zls = np.array([])

yrs = np.array([])
xrs = np.array([])
zrs = np.array([])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def iterate_frames(num, data):

    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    points = ax.scatter(data[num][0],data[num][1],data[num][2])

    return points

# User inputs the name of the video they want to analyse
# Need an error catch if an incorrect entry made by user
video_name = input("Please type the filename of the video you would like to use, including its filetype: ")
video = ("Videos/" + video_name)
cap = cv.VideoCapture(video)

# Check if camera is open
if (cap.isOpened()== False):
    print("Camera not open")
    
with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
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
            frame = cv.resize(frame, (800, 800))
            '''
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            '''
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            
            rh.append(np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y,results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z]))
            xrh.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x)
            yrh.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y)
            zrh.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z)

            # show the frame
           # cv.imshow('Running with markers. Press \'Q\' to exit.', frame)
            
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



print(rh,type(rh))

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Visualisation')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, iterate_frames, frames=len(rh), fargs=([rh]))

plt.show()

"""
viewer = napari.Viewer()
for frame in landmarks:
    viewer.add_image(frame)

viewer.layers[0].rendering = 'mip' # Maximum Intensity Projection (MIP)
viewer.dims.ndisplay = 3
napari.run()
"""