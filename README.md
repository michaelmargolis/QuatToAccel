### QuatToAccel
This is experimental code to explore techniques for calculating angular velocity or acceleration from a stream of quaternions


#### Running the software
The code uses scipi modules that require python 3.5
   pip install scikit-kinematics
   
run the read_telemetry.py file to view 20 seconds of telemetry 30 seconds from the start of the csv file

The quaternions are stored in a csv file using the following format:
   time,q0,q1,q2,q3\n
where time is the elapsed seconds since the simulated vehicle began moving_average
and q0-q3 are the quaternion elements

example showing the first two rows:
0.016,-0.000000,1.000000,0.000000,-0.001002
0.037,-0.000000,1.000000,0.000000,-0.001002


