# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        # Add additional member variables according to your need here
        #initializing variables for PID tuning
        self.integralPsiError = 0
        self.previousPsiError = 0
        self.previousXdotError = 0
        
    
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)
        # Design your controllers in the spaces below. 
        
        #Looking at the next location of the trajectory
        _, node = closestNode(X, Y, trajectory)
        
        # Choose a node that is ahead of our current node based on index
        forwardIndex = 50
        
        # condition for the completion of the path
        try:
            psiDesired = np.arctan2(trajectory[node+forwardIndex,1]-Y, \
                                    trajectory[node+forwardIndex,0]-X)
        except:
            psiDesired = np.arctan2(trajectory[-1,1]-Y, \
                                    trajectory[-1,0]-X)
        
        
        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        """
        
        #PID for delta tuning values
        kp = 1
        ki = 0.005
        kd = 0.001
        
        psiError = wrapToPi(psiDesired-psi)
        self.integralPsiError += psiError
        derivativePsiError = psiError - self.previousPsiError
        delta = kp*psiError + ki*self.integralPsiError*delT + kd*derivativePsiError/delT
                
        delta = wrapToPi(delta)
        
        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        """
        #PID for Force tuning values

        
        kp = 200
        ki = 10
        kd = 30
        # Reference value for PID to tune to
        desiredVelocity = 8
        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError
        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
        