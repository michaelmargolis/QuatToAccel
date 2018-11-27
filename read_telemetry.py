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
        self.q_array = np.zeros((5,4)) #array storing five most recent orientation quaternions
        self.time = np.zeros(1) #array storing the timestamps of each quaternion
        self.data_count = 0  # how many telemetry msgs received
        self.deltaq = np.zeros((5,4)) # array storing five most recent divided difference orientations as quaternions
        self.deltaq[0] = [1,0,0,0] # initial quat difference is the identity quaternion
        self.rotrate = np.zeros((5,3)) # array storing five most recent yaw/pitch/roll rates, in degrees per second
        self.rotacc = np.zeros((5,3)) # array storing five most recent roll/pitch/yaw accelerations, in degrees per second^2
        # for storing all the data, rather than the most recent 5:
        self.allorientation = np.zeros((1,4))
        self.allrotrate = np.zeros((1,3))
        self.allrotacc = np.zeros((1,3))

    def read(self, fname, start_time, seconds_to_read):
        # data format is:  time,q1,q2,q3,q0\n
		
        prev_time = 0  # this will be set to time of row being processed, used to calc duration
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
                    if prev_time == 0:
                        prev_time = floats[0]
                    else:
                      dur = floats[0] - prev_time
                      prev_time = floats[0]
                
		      ## Nolimits Telemetry exports quaternions in order xyzw. Standard order is wxyz
                      quat = np.array([floats[4], floats[1], floats[2], floats[3]])
		      #print(dur, quat)
                      self.process_telemetry(dur, quat, floats[0])
        #print("Pos shape",self.position.shape)
        #print("Vel shape",self.velocity.shape)
    
    def process_telemetry(self, dur, quat, time):
        # dur is time in seconds since previous quaternion
        # quat is list of the four quaternion elements
        # print(dur, quat)
        self.data_count += 1 
        #  print("count=", self.data_count)
        #  print(self.q_array[-1])
        self.q_array[:-1] = self.q_array[1:]
        self.q_array[-1] = np.array([quat[0], quat[1], quat[2], quat[3]])

        ## Keep all orientations for visualisation purposes
        self.time = np.append(self.time, time)
        self.allorientation = np.append(self.allorientation, [self.q_array[-1]], axis=0)
        #  print(self.q_array )


        if self.data_count < 2:  # need at least 2 rows to calculate velocity
            return None
		
	## Angular velocity as divided difference of quaternions
        self.deltaq[:-1] = self.deltaq[1:]
        self.deltaq[-1] = skin.quat.q_mult(skin.quat.q_inv(self.q_array[-1]),self.q_array[-2] )
		
	## Angular velocity in yaw/pitch/roll
        self.rotrate[:-1] = self.rotrate[1:] 
        self.rotrate[-1] = skin.quat.quat2seq(self.deltaq[-1],"Fick") / dur

        ## Keep all rotations and accelerations, for visualisation purposes
        self.allrotrate = np.append(self.allrotrate, [self.rotrate[-1]], axis = 0)

        if self.data_count < 3:  # need at least 3 rows to calculate acceleration
            return None
	
        ## Angular acceleration in yaw/pitch/roll
        self.rotacc[:-1] = self.rotacc[1:]
        self.rotacc[-1] = (self.rotrate[-1] - self.rotrate[-2]) / dur
		
	### These are divided differences, so are not smoothed. We could smooth each component of self.rotrate and self.rotacc separately, but a better approach would be to smooth the quaternions first, and compute the derivate of the smoothed object analytically. This can be done with the calc_angvel function from the scikit-kinematics package, but we then need to interpret the result, which is an element of the lie algebras so(3), correctly.
        
        ## Keep all rotations and accelerations, for visualisation purposes
        self.allrotacc = np.append(self.allrotacc, [self.rotacc[-1]], axis = 0)

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
        #self.animvelocity = np.zeros((frames,4)) #prepare velocity for animation
        self.animrotrate = np.zeros((frames,3))
        self.animrotacc = np.zeros((frames,3))
        #self.animrelvelocity = np.zeros((frames,4)) #prepare velocity for animation
        #self.animacceleration = np.zeros((frames,4)) #prepare acceleration for animation

        for j in range(0,len(self.animtime)):
            time = self.animtime[j]
            #print("Searching for time",time)
            idx = np.searchsorted(self.time, time)
            self.animorientation[j] = self.allorientation[idx]
            self.animrotrate[j] = self.allrotrate[idx]
            self.animrotacc[j] = self.allrotacc[idx]
            
        ## Need correct orientations to NoLimits2 world coordinates.
        rot1 = np.array( [np.cos(np.pi/4), -np.sin(np.pi/4),0,0])
        rot2 = np.array( [np.cos(np.pi/4), 0,0,np.sin(np.pi/4)])

        std2world = skin.quat.q_mult(skin.quat.q_inv(rot2),skin.quat.q_inv(rot1))
        # Quaternions are in terms of the world coordinates.
        world2std = skin.quat.q_mult(rot1,rot2)
        # or to orientate so that positive y is the identity
        #world2std = rot1
        
        temp = skin.quat.q_mult(std2world,self.animorientation)
        self.animorientation = skin.quat.q_mult(temp, world2std)

        print("Data prepared. Rendering...")

        ## For visualising orientations in 3d
        skin.view.orientation(self.animorientation,title_text="Orientation",out_file = "orientation.mp4", deltaT=20)
        
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.axes3d as p3
        import matplotlib.animation as animation

        ## Function for exporting linegraphs for velocity and acceleration magnitudes.
        def export(time,data,filename,title,ylabel):

            fig, ax = plt.subplots()
            xdata, ydata = [], []
            ln, = plt.plot([], [], 'b-', animated=True)
           # plt.yscale('log')
            plt.title(title)
            plt.ylabel(ylabel)
            ax.set_ylim(min(data) , max(data))

            def init():
                ax.set_xlim(start_time, start_time + secs_to_read)
                #ax.set_ylim(0, 100)
                #ax.set_yscale("log",nonposy='clip')
                #plt.yscale('log')
                #plt.title(title)
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

        
        yawrate = self.animrotrate[:,0]
        pitchrate = self.animrotrate[:,1]
        rollrate = self.animrotrate[:,2]

        yawacc = self.animrotacc[:,0]
        pitchacc = self.animrotacc[:,1]
        rollacc = self.animrotacc[:,2]
           
            
        export(self.animtime,rollrate,"RollRate.mp4", "Roll Rate","Degrees / second")
        export(self.animtime,pitchrate,"PitchRate.mp4", "Pitch Rate","Degrees / second")
        export(self.animtime,yawrate,"YawRate.mp4", "Yaw Rate","Degrees / second")

        export(self.animtime,rollacc,"RollAcc.mp4", "Roll Acceleration","Degrees / second^2")
        export(self.animtime,pitchacc,"PitchAcc.mp4", "Pitch Acceleration","Degrees / second^2")
        export(self.animtime,yawacc,"YawAcc.mp4", "Yaw Acceleration","Degrees / second^2") 


start_time = 0  #process telemetry data that is greater than or equal to this time in seconds
secs_to_read = 90  # number of seconds of telemetry data to process

if __name__ == "__main__":
    telemetry = Telemetry()
    telemetry.read('telemetry.csv', start_time, secs_to_read )
    
    #skin.view.orientation(telemetry.velocity,title_text="Velocity",out_file = "velocity.mp4", deltaT=20)
    #skin.view.orientation(telemetry.acceleration,title_text="Acceleration",out_file = "acceleration.mp4", deltaT=20)
    #telemetry.visualise()
