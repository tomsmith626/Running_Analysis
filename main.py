import cv2 as cv
import numpy as np
import mediapipe as mp
import napari
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# hip, knee, ankle, shoulder, wrist, heel, footindex
coordinates = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
coordinate_names = ["RIGHT_HIP", "LEFT_HIP","RIGHT_KNEE", "LEFT_KNEE", "RIGHT_ANKLE", "LEFT_ANKLE","RIGHT_SHOULDER", "LEFT_SHOULDER",
                    "RIGHT_WRIST", "LEFT_WRIST", "RIGHT_HEEL", "LEFT_HEEL","RIGHT_FOOT_INDEX","LEFT_FOOT_INDEX"]

def get_joint_coords(results, coordinates, coordinate_names):
    for i, joint in enumerate(coordinate_names):
        # feel like the below line could easily be shortened, I'm calling the reults for one thing three times
        coordinates[i].append(np.array([results.pose_landmarks.landmark[getattr(mp_holistic.PoseLandmark,joint)].x,results.pose_landmarks.landmark[getattr(mp_holistic.PoseLandmark,joint)].y,results.pose_landmarks.landmark[getattr(mp_holistic.PoseLandmark,joint)].z]))
    return coordinates

def iterate_frames(num, data):
    """
    This function is used in the animation to produce the graphs in each frame.
    num is the frame number we are on
    data is a list of lists of np.arrays - each sub-list is a joint and each np.array is the position of that joint at a given frame
    """

    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    
    for i, joint in enumerate(data):
        if i == 6 or i == 7:
            points = ax.scatter(joint[num][0],joint[num][1],joint[num][2],c="red")
        else:
            points = ax.scatter(joint[num][0],joint[num][1],joint[num][2],c="blue")
        
    #trying to plot lines between joints that need lines
    points = ax.plot3D(np.array([data[0][num][0],data[1][num][0]]),np.array([data[0][num][1],data[1][num][1]]),np.array([data[0][num][2],data[1][num][2]]))
    points = ax.plot3D(np.array([data[0][num][0],data[2][num][0]]),np.array([data[0][num][1],data[2][num][1]]),np.array([data[0][num][2],data[2][num][2]]))
    points = ax.plot3D(np.array([data[1][num][0],data[3][num][0]]),np.array([data[1][num][1],data[3][num][1]]),np.array([data[1][num][2],data[3][num][2]]))
    points = ax.plot3D(np.array([data[2][num][0],data[4][num][0]]),np.array([data[2][num][1],data[4][num][1]]),np.array([data[2][num][2],data[4][num][2]]))
    points = ax.plot3D(np.array([data[6][num][0],data[7][num][0]]),np.array([data[6][num][1],data[7][num][1]]),np.array([data[6][num][2],data[7][num][2]]))
    points = ax.plot3D(np.array([data[6][num][0],data[0][num][0]]),np.array([data[6][num][1],data[0][num][1]]),np.array([data[6][num][2],data[0][num][2]]))
    points = ax.plot3D(np.array([data[7][num][0],data[1][num][0]]),np.array([data[7][num][1],data[1][num][1]]),np.array([data[7][num][2],data[1][num][2]]))


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

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            
            coordinates = get_joint_coords(results, coordinates,coordinate_names)

            # show the frame
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


# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the axes properties - this should probably be calculated from max & min values from opencv data
ax.set_xlim3d([-1, 1])
ax.set_xlabel('X')

ax.set_ylim3d([-1, 1])
ax.set_ylabel('Y')

ax.set_zlim3d([-1, 1])
ax.set_zlabel('Z')

ax.set_title('3D Visualisation')

"""
Creating the Animation object. my understanding:
frames = number of frames before the thing repeats itself
fargs = the arguments that are sent to the function that returns the next image to animate (frame is also sent which just equals the number frame we are on) 
        currently only sending rh data.
"""
line_ani = animation.FuncAnimation(fig, iterate_frames, frames=len(coordinates[0]), fargs=([coordinates]))

plt.show()

# saving to m4 using ffmpeg writer 
writervideo = animation.FFMpegWriter(fps=60) 
line_ani.save('increasingStraightLine.mp4', writer=writervideo) 
plt.close() 

"""
viewer = napari.Viewer()
for frame in landmarks:
    viewer.add_image(frame)

viewer.layers[0].rendering = 'mip' # Maximum Intensity Projection (MIP)
viewer.dims.ndisplay = 3
napari.run()
"""