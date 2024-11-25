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
        self.previous_err_F = 0
        self.previous_err_delta = 0
        self.integral_F = 0
        self.integral_delta = 0
        
    #PID controller for longitudinal dynamics
    def pid_F(self,err,p,i,d,delT):
    
        proportional = p*err
        self.integral_F+= err*delT
        integral = i*self.integral_F
        derivative = d*(err - self.previous_err_F)/delT
        self.previous_err_F = err
        
        return (proportional+integral+derivative)
            
    #PID controller for ltaeral dynamics
    def pid_delta(self,err,p,i,d,delT):
    
        proportional = p*err
        self.integral_delta += err*delT
        integral = i*self.integral_delta
        derivative = d*(err - self.previous_err_delta)/delT
        self.previous_err_delta = err
        
        return (proportional+integral+derivative)
    
    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        
        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        # Design your controllers in the spaces below. 
        
        #Looking at the next location of the trajectory
        node = closestNode(X,Y,trajectory) 
        
        #The next position to look at is 90 nodes ahead
        look_ahead = 90
        
        # condition for the completion of the path
        if(node[1]+look_ahead)>= 8203:
            look_ahead = 20
        
        #next location points in the trajectory
        X_next = trajectory[node[1]+look_ahead,0]
        Y_next = trajectory[node[1]+look_ahead,1]
        
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        """
        
        #PID for delta tuning values
        K_p_delta = 2
        K_d_delta = 0.05
        K_i_delta = 6
        
        
        psi_desired = np.arctan2((Y_next - Y),(X_next - X))
        err_delta = wrapToPi(psi_desired - psi)
        
        delta = self.pid_delta(err_delta,K_p_delta,K_d_delta,K_i_delta,delT)
        
        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        """
        #PID for Force tuning values
        K_p_F = 2
        K_d_F = 0.0
        K_i_F = 0.0
        
        err_distance = np.sqrt((X - X_next)**2 + (Y-Y_next)**2)
        err_velocity = err_distance/delT
        
        F = self.pid_F(err_velocity,K_p_F,K_d_F,K_i_F,delT)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
        