"""
read_telemetry.py

rotation orientation as quaternions are read from an input file
quaternions are smoothed using window size of five

code requires python 3.5 or later
tested with anaconda and these packages :
   pip install scikit-kinematics
"""



import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    print("this version needs python 3.5 or later")
    exit()

import scipy.signal
import skinematics as skin
import numpy as np
import traceback

class Telemetry():

    def __init__(self):
        self.q_array = np.zeros((5,4)) #array storing most recent five quaternions
        self.time = np.zeros(1) #array storing the timestamps of each quaternion
        self.data_count = 0  # how many telemetry msgs received
        self.orientation = np.zeros((1,4)) #array to store all position quaternions
        #self.velocity = np.zeros((6,4)) #array to store all velocity quaternions
        #self.acceleration = np.zeros((6,4)) #array to store all acceleration quaternions

    def read(self, fname, start_time, seconds_to_read):
        # data format is:  time,q0,q1,q2,q3\n
        with open(fname, 'r') as f:
            data = f.read().split('\n')
            for row in data:
                fields = row.split(',')
                floats = []
                for elem in fields:
                    try:
                        floats.append(float(elem))
                    except ValueError:
                        print("error in csv file", floats)
                # print(len(floats))
                if floats[0] >= start_time and floats[0] <= start_time + seconds_to_read:
                    self.time = np.append(self.time, floats[0])
                    self.orientation = np.append(self.orientation, [floats[1:5]], axis=0)
                      
                      #self.process_telemetry(dur, floats[1:5], floats[0])
        #print("Pos shape",self.position.shape)
        #print("Vel shape",self.velocity.shape)
    
    def process_telemetry(self):
        data_count = len(self.orientation)
        time_delta = self.time[1:] - self.time[:-1] # time delta
        ## Divided difference for dq_dt
        dq_dt = np.zeros((data_count,4))
        dq_dt[1:] = (self.orientation[1:] - self.orientation[:-1]) / time_delta[:,None]

        ## Divided difference for d2q_dt2
        d2q_dt2 = np.zeros((data_count,4))
        d2q_dt2[2:] = (dq_dt[2:] - dq_dt[:-2]) / time_delta[1:,None]

        ## Calculate velocity
        self.velocity = 2 * skin.quat.q_mult(dq_dt, skin.quat.q_inv(self.orientation))

        ## Calculate acceleration
        temp = skin.quat.q_mult(dq_dt, skin.quat.q_inv(self.orientation))
        self.acceleration = 2 * (skin.quat.q_mult(d2q_dt2, skin.quat.q_inv(self.orientation)) - skin.quat.q_mult(temp,temp))
        

    def visualise(self):
        # Coastervideo is 30 fps
        fps = 30
        # we want one frame every
        delay = 1/30  # seconds

        # Need to drop some frames
        endtime = self.time[-1] #
        self.animtime = np.arange(0,endtime, delay) #times for animation
        frames = len(self.animtime)
        print("Number of frames to produce",frames)
        self.animorientation = np.zeros((frames,4)) #prepare orientations for animation
        self.animvelocity = np.zeros((frames,4)) #prepare velocity for animation
        self.animacceleration = np.zeros((frames,4)) #prepare acceleration for animation

        for j in range(0,len(self.animtime)):
            time = self.animtime[j]
            print("Searching for time",time)
            idx = np.searchsorted(self.time, time)
            self.animorientation[j] = self.orientation[idx]
            self.animvelocity[j] = self.velocity[idx]
            self.animacceleration[j] = self.acceleration[idx]
            

        print("Data prepared. Rendering...")
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        def export(time,data,filename,title):

            fig, ax = plt.subplots()
            xdata, ydata = [], []
            ln, = plt.plot([], [], 'b-', animated=True)

            def init():
                ax.set_xlim(start_time, start_time + secs_to_read)
                ax.set_ylim(0, 100)
                plt.title(title)
                return ln,

            def update(frame):
                xdata.append(time[frame])
                ydata.append(data[frame])
                ln.set_data(xdata, ydata)
                return ln,

            anim = animation.FuncAnimation(fig, update, frames=len(time), interval=delay*1000,
                    init_func=init, blit=True)
            
            FFwriter=animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
            anim.save(filename, writer=FFwriter)

            plt.show()

        
        vel = np.linalg.norm(self.animvelocity,axis=1)
        acc = np.linalg.norm(self.animacceleration,axis=1)


        skin.view.orientation(self.animorientation,title_text="Orientation",out_file = "orientation.mp4", deltaT=delay*1000)
        skin.view.orientation(self.animvelocity,title_text="Velocity",out_file = "velocity.mp4", deltaT=delay*1000)
        skin.view.orientation(self.animacceleration,title_text="Acceleration",out_file = "acceleration.mp4", deltaT=delay*1000)
        
        export(self.animtime,vel,"Velocity Magnitude.mp4", "Velocity magnitude")
        export(self.animtime,acc,"Acceleration Magnitude.mp4", "Acceleration magnitude")

        

start_time = 0  #process telemetry data that is greater than or equal to this time in seconds
secs_to_read = 90  # number of seconds of telemetry data to process

if __name__ == "__main__":
    telemetry = Telemetry()
    telemetry.read('telemetry.csv', start_time, secs_to_read )
    telemetry.process_telemetry()
    telemetry.visualise()
