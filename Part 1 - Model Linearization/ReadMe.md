## Exercise 1 - Model Linearization ##

As mentioned in class, model linearization is always the first step for non-linear control. During this assignment, you will approximate the given model with a linear model.


## Exercise 2 - Controller synthesis in Simulation ##

The driver functions that control the car take the desired steering angle δ and a throttle input - ranging from 0 to 1 - which is derived from the desired longitudinal force F.
For this question, you have to design a PID longitudinal controller and a PID lateral controller for the vehicle. A PID is an error-based controller that requires tuning proportional, integral, and derivative gains. As a PID allows us to minimize the error between a set point and process variable without deeper knowledge of the system model, we will not need our result from Exercise 1 (though it will be useful in future project parts).
Design the two controllers in your controller.py. You can make use of Webots’ builtin code editor, or use your own.
