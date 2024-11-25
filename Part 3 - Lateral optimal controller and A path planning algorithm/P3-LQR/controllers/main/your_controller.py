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
        
        self.previous_err_F = 0
        self.integral_F = 0
        self.prev_e1 = 0
        
        # Add additional member variables according to your need here.
        
    def pid_F(self,err,p,i,d,delT):
    
        proportional = p*err
        self.integral_F+= err*delT
        integral = i*self.integral_F
        derivative = d*(err - self.previous_err_F)/delT
        self.previous_err_F = err
        
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
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        
        node = closestNode(X,Y,trajectory) 
        
        #The next position to look at is 90 nodes ahead
        look_ahead = 70
        
        # condition for the completion of the path
        if(node[1]+look_ahead)>= 8203:
            look_ahead = 20
        
        #next location points in the trajectory
        X_next = trajectory[node[1]+look_ahead,0]
        Y_next = trajectory[node[1]+look_ahead,1]
        

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        
        psi_desired = wrapToPi(np.arctan2(Y_next - Y,X_next - X))
        err_delta = wrapToPi(psi-psi_desired)
        e1 = node[0]
        e1_dot = (e1 - self.prev_e1)/delT
        self.prev_e1 = e1
        e2 = err_delta
        e2_dot = psidot
        
        # State inputs
        err = np.array([e1,e1_dot,e2,e2_dot])
        x = err.T
       
        #Error Matrix A
        A_cont = np.array([[0,1,0,0],
              [0,(-4*self.Ca/(self.m*xdot)),(4*self.Ca/self.m),(-2*self.Ca*(self.lf -self.lr)/(self.m*xdot))],
              [0,0,0,1],
              [0,(-2*self.Ca*(self.lf - self.lr)/(self.Iz*xdot)),(2*self.Ca*(self.lf - self.lr)/self.Iz),(-2*self.Ca*(self.lf**2 + self.lr**2)/(self.Iz*xdot))]])
        
        #Error Matrix B
        B_cont = np.array([[0],
                      [(2*self.Ca/m)],
                      [0],
                      [(2*self.Ca*self.lf/self.Iz)]])
        
        C_cont = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1]])
        D_cont = np.array([[0],[0],[0],[0]])
        
        A,B,_,_,_ = signal.cont2discrete((A_cont,B_cont,C_cont,D_cont),dt = delT)
        
        Q = np.diag([0.05,0.05,0.01,25])
        R = np.diag([1])
        
        S = np.matrix(linalg.solve_discrete_are(A,B,Q,R))
        K = -np.matrix(linalg.inv(B.T@S@B+R)@(B.T@S@A))
        upd_delta = ((K@x))
        delta = np.double(upd_delta)

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        .
        .
        .
        .
        .
        .
        .
        .
        .
        """
        
        K_p_F = 3
        K_d_F = 0.0
        K_i_F = 4
        
        err_distance = np.sqrt((X - X_next)**2 + (Y-Y_next)**2)
        err_velocity = err_distance/delT
        
        F = self.pid_F(err_velocity,K_p_F,K_d_F,K_i_F,delT)

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
