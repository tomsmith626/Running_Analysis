import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def iterate_frames(num, data):

    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()
    points = ax.scatter(data[num][0],data[num][1],data[num][2])

    return points

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)


data = [np.array([0.5,0.5,0.5]),np.array([0.4,0.4,0.4]),np.array([0.3,0.3,0.3]),np.array([0.2,0.2,0.2])]

points = [ax.scatter(dat[0], dat[1], dat[2]) for dat in data]

print(data[1], type(data[1]), type(data))

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, iterate_frames, frames=4, fargs=([data]))

plt.show()