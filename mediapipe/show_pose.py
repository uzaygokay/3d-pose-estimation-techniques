import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso_r = [[0, 1] , [1, 7]]
    torso_l = [[7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso_r, torso_l, arml, armr, legr, legl]
    colors = ['red', 'purple', 'blue', 'green', 'black', 'orange']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #for framenum, kpts3d in enumerate(p3ds):
    #    if framenum%2 == 0: continue #skip every 2nd frame
    #    for bodypart, part_color in zip(body, colors):
    #        for _c in bodypart:
                #ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)


    for framenum, kpts3d in enumerate(p3ds):
        if framenum%2 == 0: continue #skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                x = kpts3d[_c[0],0]
                y = kpts3d[_c[0],1]
                z = kpts3d[_c[0],2]
                x2 = kpts3d[_c[1],0]
                y2 = kpts3d[_c[1],1]
                z2 = kpts3d[_c[1],2]
                ax.plot(xs = [z, z2], ys = [x, x2], zs = [-y, -y2], linewidth = 4, c = part_color)


        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-20, 20)
        ax.set_xlabel('x')
        ax.set_ylim3d(-20, 20)
        ax.set_ylabel('y')
        ax.set_zlim3d(-20, 20)
        ax.set_zlabel('z')

        
        plt.pause(0.025)
        ax.cla()


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)