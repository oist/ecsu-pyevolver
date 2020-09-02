from scipy.special import expit
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.random import RandomState


def center_cross(weights):
    """
    Set network bias terms so that the null-(hyper)surfaces of each neuron intersect at their exact centers of symmetry,
    ensuring that each neuron's activation function is centered over the range of net inputs that it receives.
    :param weights: weights of the network
    :return:
    """
    theta = -np.sum(weights, axis=1) / 2
    return theta


@dataclass
class BrainCTRNN:
    """
    Initialize a fully connected ctrnn of size N (num_neurons the following attributes:

    states   = 'state of each neuron' at current time point i
    taus     = 'time constant (tau > 0)'
    biases   = 'the bias terms'
    gains    = 'gain' (makes neurons highly sensitive to their input, primarily for motor or sensory nodes)
                Preferably between [1,5] and just > 1 for neurons connected to sensory input
                or motor output.
    weights  = 'fixed strength of the connection from jth to ith neuron', Weight Matrix
    sigma    = 'the sigmoid function / standard logistic activation function' 1/(1+np.exp(-x))
    input    = 'constant external input' at current time point i

    :param random_seed: random seed to be initialized
    :param num_neurons: number of neurons in the network
    :param step_size: step size for the update function
    :param tau_range, gain_range, bias_range, weight_range: parameter ranges
    :param state_range: the range of initial neural state when initializing the network
    :param dy_dt, input, output, states: network derivative, input, output and state at a given time step
    :param taus, gains, weights, biases: network parameters as described above
    """

    random_seed: int = 0
    num_neurons: int = 1    
    step_size: float = 0.01
    tau_range: Tuple = (1, 1)
    gain_range: Tuple = (1, 1)
    bias_range: Tuple = (0, 0)
    weight_range: Tuple = (0, 0)
    state_range: Tuple = (0, 1)
    dy_dt: np.ndarray = None
    input: np.ndarray = None
    output: np.ndarray = None
    taus: np.ndarray = None
    gains: np.ndarray = None
    weights: np.ndarray = None
    biases: np.ndarray = None
    states: np.ndarray = None

    def __post_init__(self):
        self.random_state = RandomState(self.random_seed)
        if self.dy_dt is None:
            self.dy_dt = np.zeros(self.num_neurons)  # initialized with zeros
        if self.input is None:
            self.input = np.zeros(self.num_neurons)  # initialized with zeros
        if self.states is None:
            self.states = self.rand_param(self.state_range)  # random
        if self.taus is None:    
            self.taus = self.rand_param(self.tau_range)  # random
        if self.gains is None:    
            self.gains = self.rand_param(self.gain_range)  # random
        if self.biases is None:    
            self.biases = self.rand_param(self.bias_range)  # random
        if self.weights is None:    
            self.weights = self.random_state.uniform(
                self.weight_range[0], self.weight_range[1], 
                (self.num_neurons, self.num_neurons)
            )  # random
        if self.weights.ndim == 1:
            self.weights = self.weights.reshape(self.num_neurons, -1)
            # auto reshape in case of flat array

    def rand_param(self, param_range):
        return self.random_state.uniform(param_range[0], param_range[1], self.num_neurons)

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
        self.random_state = RandomState(self.random_seed)

    def randomize_state(self):
        # To start the simulation it is often useful to randomize initial neuron activation around 0
        self.states = self.rand_param(self.state_range)

    def euler_step(self):
        # Compute the next state of the network given its current state and the simple euler equation
        # update the state of all neurons
        self.dy_dt = np.multiply(1 / self.taus,
                                 - self.states + np.dot(self.output, self.weights) + self.input) * self.step_size
        self.states += self.dy_dt
        # update the outputs of all neurons
        self.compute_output()

    def compute_output(self):
        self.output = expit(np.multiply(self.gains, self.states + self.biases))

    def get_state(self):
        return self.states

    def get_output(self):
        return self.output

    def __str__(self):
        return "taus {}\n gains {}\n weights {}\n biases {}".format(self.taus, self.gains, self.weights, self.biases)


"""
Consider to implement:
- RK4 update step in addition to Euler step
- fast sigmoid with pre-computed table
"""
