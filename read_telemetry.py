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
        self.position = np.zeros((1,4)) #array to store all position quaternions, slow as we append at each stage.
        self.velocity = np.zeros((6,4)) #array to store all velocity quaternions, slow as we append at each stage. Can't calculate the first 5
        self.acceleration = np.zeros((6,4)) #array to store all acceleration quaternions, slow as we append at each stage. Can't calculate the first 5

    def read(self, fname, start_time, seconds_to_read):
        # data format is:  time,q0,q1,q2,q3\n
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
                      #print(dur, floats[1:5])
                      self.process_telemetry(dur, floats[1:5], floats[0])
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
        ## Keep all telemetry for visualisation purposes
        self.time = np.append(self.time, time)
        self.position = np.append(self.position, [self.q_array[-1]], axis=0)
        #  print(self.q_array )
        if self.data_count < 6:  # need at least 5 rows to calculate velocity
            return None
        a = self.calc_angvel(self.q_array, dur)
        #print("output from calc_angvel=\n",  a)
        self.velocity = np.append(self.velocity, [a[-1]], axis=0)
        b = self.calc_angacc(self.q_array, dur)
        self.acceleration = np.append(self.acceleration, [b[-1]], axis=0)
        # TODO .....
        

    def calc_angvel(self, q, interval, winSize=5, order=2):
        '''
        Take a quaternion, and convert it into the
        corresponding angular velocity

        Parameters
        ----------
        q : array, shape (N,[3,4])
            unit quaternion vectors.
            N is at least winSize
        rate : float
            sampling rate (in [Hz])
        winSize : integer
            window size for the calculation of the velocity.
            Has to be odd.
        order : integer
            Order of polynomial used by savgol to calculate the first derivative

        Returns
        -------
        angvel : array, shape (3,) or (N,3)
            angular velocity [rad/s].

        Notes
        -----
        The angular velocity is given by

          .. math::
            \\omega = 2 * \\frac{dq}{dt} \\circ q^{-1}

        Examples
        --------
        >>> rate = 1000
        >>> t = np.arange(0,10,1/rate)
        >>> x = 0.1 * np.sin(t)
        >>> y = 0.2 * np.sin(t)
        >>> z = np.zeros_like(t)
        array([[ 0.20000029,  0.40000057,  0.        ],
               [ 0.19999989,  0.39999978,  0.        ],
               [ 0.19999951,  0.39999901,  0.        ]])
                .......
        '''
        
        if np.mod(winSize, 2) != 1:
            raise ValueError('Window size must be odd!')
        
        numCols = 4 ##### fixme   q.shape[1]
        if numCols != 4:
            raise TypeError('quaternions must have 4 columns')
        # print("rotation axis=", rotationAxis)
        ## Ok, the filter (or "smoother" to be more precise) should use future data to determine current data (with a window of 5 we need
        ## 2 future quaternions). Instead, we are using the previous 5, which might not give the best approximation.
        ## A better mathematical understanding would be needed.
        dq_dt = scipy.signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=1, delta=interval, axis=0)
        ## Also filter q_inv
        q_inv_filter = scipy.signal.savgol_filter(skin.quat.q_inv(q), window_length=winSize, polyorder=order, deriv=0, delta=interval, axis=0)
        
        angVel = 2 * skin.quat.q_mult(dq_dt, q_inv_filter)
        return angVel

    def calc_angacc(self, q, interval, winSize=5, order=3):
        '''
        Calculate the acceleration from a sequence of orientations, all described as quaternions.

        Parameters
        ----------
        q : array, shape (N,[3,4])
            unit quaternion vectors.
            N is at least winSize
        rate : float
            sampling rate (in [Hz])
        winSize : integer
            window size for the calculation of the velocity.
            Has to be odd.
        order : integer
            Order of polynomial used by savgol to calculate the first derivative

        Returns
        -------
        angacc : array, shape (3,) or (N,3)
            angular acceleration (N,3)

        Notes
        -----
        The angular acceleration is given by

          .. math::
            \\omega = 2 * \\frac{d^2q}{dt^2} \\circ q^{-1} - (\\frac{dq}{dt} \\circ q^{-1})^2

        Examples
        --------
        >>> rate = 1000
        >>> t = np.arange(0,10,1/rate)
        >>> x = 0.1 * np.sin(t)
        >>> y = 0.2 * np.sin(t)
        >>> z = np.zeros_like(t)
        array([[ 0.20000029,  0.40000057,  0.        ],
               [ 0.19999989,  0.39999978,  0.        ],
               [ 0.19999951,  0.39999901,  0.        ]])
                .......
        '''
        
        if np.mod(winSize, 2) != 1:
            raise ValueError('Window size must be odd!')
        
        numCols = 4 ##### fixme   q.shape[1]
        if numCols != 4:
            raise TypeError('quaternions must have 4 columns')
        # print("rotation axis=", rotationAxis)
        ## Ok, the filter (or "smoother" to be more precise) should use future data to determine current data (with a window of 5 we need
        ## 2 future quaternions). Instead, we are using the previous 5, which might not give the best approximation.
        ## A better mathematical understanding would be needed.
        dq_dt = scipy.signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=1, delta=interval, axis=0)
        d2q_dt2 = scipy.signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=2, delta=interval, axis=0)
        ## Also filter q_inv
        q_inv_filter = scipy.signal.savgol_filter(skin.quat.q_inv(q), window_length=winSize, polyorder=order, deriv=0, delta=interval, axis=0)
        
        temp = skin.quat.q_mult(dq_dt, q_inv_filter)
        angAcc = 2 * (skin.quat.q_mult(d2q_dt2, q_inv_filter) - skin.quat.q_mult(temp,temp))
        return angAcc

    def visualise(self):
        
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

            anim = animation.FuncAnimation(fig, update, frames=len(time), interval=20,
                    init_func=init, blit=True)
            ## 20ms between frames gives 50 fps
            FFwriter=animation.FFMpegWriter(fps=50, extra_args=['-vcodec', 'libx264'])
            anim.save(filename, writer=FFwriter)

            plt.show()

        
        time = self.time
        vel = np.linalg.norm(self.velocity,axis=1)
        acc = np.linalg.norm(self.acceleration,axis=1)

        export(time,vel,"Velocity Magnitude.mp4", "Velocity magnitude")
        export(time,acc,"Acceleration Magnitude.mp4", "Acceleration magnitude")

start_time = 0  #process telemetry data that is greater than or equal to this time in seconds
secs_to_read = 90  # number of seconds of telemetry data to process

if __name__ == "__main__":
    telemetry = Telemetry()
    telemetry.read('telemetry.csv', start_time, secs_to_read )
    skin.view.orientation(telemetry.position,title_text="Orientation",out_file = "orientation.mp4", deltaT=20)
    skin.view.orientation(telemetry.velocity,title_text="Velocity",out_file = "velocity.mp4", deltaT=20)
    skin.view.orientation(telemetry.acceleration,title_text="Acceleration",out_file = "acceleration.mp4", deltaT=20)
    telemetry.visualise()
