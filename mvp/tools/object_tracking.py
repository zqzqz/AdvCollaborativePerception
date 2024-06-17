import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


class KalmanFilter(object):
    """docstring for KalmanFilter"""

    def __init__(self, dt=1,stateVariance=1,measurementVariance=1, 
                                                        method="Velocity" ):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
    
    """init function to initialise the model"""
    def initModel(self, measurement): 
        if self.method == "Accerelation":
            self.U = 1
        else: 
            self.U = 0
        self.A = np.matrix( [[1 ,self.dt, 0, 0], [0, 1, 0, 0], 
                                        [0, 0, 1, self.dt],  [0, 0, 0, 1]] )

        self.B = np.matrix( [[self.dt**2/2], [self.dt], [self.dt**2/2], 
                                                                [self.dt]] )
        
        self.H = np.matrix( [[1,0,0,0], [0,0,1,0]] ) 
        self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance*np.identity(
                                                            self.H.shape[0]))
        
        self.Q = np.matrix( [[self.dt**4/4 ,self.dt**3/2, 0, 0], 
                            [self.dt**3/2, self.dt**2, 0, 0], 
                            [0, 0, self.dt**4/4 ,self.dt**3/2],
                            [0, 0, self.dt**3/2,self.dt**2]])
        
        self.erroCov = self.P
        self.state = np.matrix([[measurement[0]],[1],[measurement[1]],[1]])


    """Predict function which predicst next state based on previous state"""
    def predict(self):
        self.predictedState = self.A*self.state + self.B*self.U
        self.predictedErrorCov = self.A*self.erroCov*self.A.T + self.Q
        temp = np.asarray(self.predictedState)
        return temp[0], temp[2]

    """Correct function which correct the states based on measurements"""
    def correct(self, currentMeasurement):
        self.kalmanGain = self.predictedErrorCov*self.H.T*np.linalg.pinv(
                                self.H*self.predictedErrorCov*self.H.T+self.R)
        self.state = self.predictedState + self.kalmanGain*(currentMeasurement
                                               - (self.H*self.predictedState))
        
        self.erroCov = (np.identity(self.P.shape[0]) - 
                                self.kalmanGain*self.H)*self.predictedErrorCov


class Tracks(object):
    """docstring for Tracks"""
    def __init__(self, detection, trackId):
        super(Tracks, self).__init__()
        self.KF = KalmanFilter()
        self.KF.initModel(detection)
        self.prediction = detection.reshape(1,2)
        self.trackId = trackId
        self.skipped_frames = 0

    def predict(self,detection):
        self.prediction = np.array(self.KF.predict()).reshape(1,2)
        self.KF.correct(np.matrix(detection).reshape(2,1))


class Tracker(object):
    """docstring for Tracker"""
    def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
        super(Tracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.max_trace_length = max_trace_length
        self.trackId = 0
        self.tracks = []

    def update(self, detections):
        if len(self.tracks) == 0:
            for i in range(detections.shape[0]):
                track = Tracks(detections[i], self.trackId)
                self.trackId +=1
                self.tracks.append(track)

        N = len(self.tracks)
        M = len(detections)
        cost = []
        predictions = []
        for i in range(N):
            predictions.append(self.tracks[i].prediction[0].tolist())
            diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
            cost.append(diff)
        cost = np.array(cost)

        row, col = linear_sum_assignment(cost)
        assignment = [-1]*N
        for i in range(len(row)):
            assignment[row[i]] = col[i]

        assigned_tracks = []
        assigned_detections = []

        for i in range(len(assignment)):
            if assignment[i] != -1:
                if (cost[i][assignment[i]] > self.dist_threshold):
                    assignment[i] = -1
            if assignment[i] < 0:
                self.tracks[i].skipped_frames += 1
            else:
                assigned_tracks.append(i)
                assigned_detections.append(assignment[i])

        unassigned_tracks = list(set([i for i in range(len(self.tracks))]) - set(assigned_tracks))
        unassigned_detections = list(set([i for i in range(len(detections))]) - set(assigned_detections))
        error_tracks = []
        for i in unassigned_tracks:
            error_tracks.append(np.array(self.tracks[i].prediction[0]))
        error_detections = []
        for j in unassigned_detections:
            error_detections.append(detections[j])

        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frame_skipped:
                del_tracks.append(i)

        if len(del_tracks) > 0:
            del_tracks.sort(reverse=True)
            for i in range(len(del_tracks)):
                del self.tracks[i]
                del assignment[i]

        for i in range(len(assignment)):
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].predict(detections[assignment[i]])

        for i in range(len(detections)):
            if i not in assignment:
                track = Tracks(detections[i], self.trackId)
                self.trackId +=1
                self.tracks.append(track)
                assignment.append(i)

        return np.array(predictions), assignment, error_tracks, error_detections
