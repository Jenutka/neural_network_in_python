#!/usr/bin/env python3

# ------------------------------ #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# ------------------------------ #
class AniPlot():

    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-3.5, 3.5), ylim=(-5, 2))
        self.line, = self.ax.plot([], [], lw=2)


    def set_data(self,data):
        self.data = data

    def ani_init(self):
        self.line.set_data([], [])
        return self.line

    def ani_update(self, i):
        x = self.data[i][0]
        y = self.data[i][1]
        self.line.set_data(x, y)
        return self.line


    def animate(self):
        self.anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.ani_init, frames=4, interval=20, blit=False)
        plt.show()

# ------------------------------ #

data = [
[[0,0,1,0],[0,-1,-2,-3]],
[[0,0,0,0.1],[0,-1,-3,-4]],
[[0,0,0.5,0],[0,-1,-2.5,-3.5]],
[[0,0,1,2],[0,-1,-2,-2.5]]
        ]
myani = AniPlot()
myani.set_data(data)
myani.animate()
