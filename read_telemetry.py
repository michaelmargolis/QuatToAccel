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
        self.data_count = 0  # how many telemetry msgs received

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
                      self.process_telemetry(dur, floats[1:5])

    
    def process_telemetry(self, dur, quat):
        # dur is time in seconds since previous quaternion
        # quat is list of the four quaternion elements
        # print(dur, quat)
        self.data_count += 1 
        #  print("count=", self.data_count)
        #  print(self.q_array[-1])
        self.q_array[:-1] = self.q_array[1:]
        self.q_array[-1] = np.array([quat[0], quat[1], quat[2], quat[3]])
        #  print(self.q_array )
        if self.data_count < 6:  # need at least 5 rows to calculate velocity
            return None
        a = self.calc_angvel(self.q_array, dur, 0)
        print("output from calc_angvel=\n",  a)
        # TODO .....


    def calc_angvel(self, q, interval, rotationAxis, winSize=5, order=2):
        '''
        Take a quaternion, and convert it into the
        corresponding angular velocity

        Parameters
        ----------
        q : array, shape (N,[3,4])
            unit quaternion vectors.
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
        dq_dt = scipy.signal.savgol_filter(q, window_length=winSize, polyorder=order, deriv=1, delta=interval, axis=rotationAxis)
        angVel = 2 * skin.quat.q_mult(dq_dt, skin.quat.q_inv(q))
        return angVel


start_time = 30  #process telemetry data that is greater than or equal to this time in seconds
secs_to_read = 20  # number of seconds of telemetry data to process

if __name__ == "__main__":
    telemetry = Telemetry()
    telemetry.read('telemetry.csv', start_time, secs_to_read )

