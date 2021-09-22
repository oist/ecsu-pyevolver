import numpy as np
import matplotlib.pyplot as plt
from pyevolver.ctrnn import BrainCTRNN


def plot_phase_space(ctrnn_brain, net_history=None):
    """ Plot the phase portrait
    We'll use matplotlib quiver function, which wants as arguments the grid of x and y coordinates,
    and the derivatives of these coordinates.
    In the plot we see the locations of stable and unstable equilibria,
    and can eyeball the trajectories that the system will take through
    the state space by following the arrows.
    """
    # Define the sample space (plotting ranges)
    # ymin = np.amin(net_history)
    # ymax = np.amax(net_history)
    ymin = -10
    ymax = 10
    y1 = np.linspace(ymin, ymax, 30)
    y2 = np.linspace(ymin, ymax, 30)
    Y1, Y2 = np.meshgrid(y1, y2)
    dim_y = y1.shape[0]

    # calculate the state space derivatives across our sample space
    changes_y1 = np.zeros([dim_y, dim_y])
    changes_y2 = np.zeros([dim_y, dim_y])

    def compute_derivatives(states):
        return ctrnn_brain.step_size * \
            np.multiply(
                1 / ctrnn_brain.taus,
                - states + np.dot(ctrnn_brain.output, ctrnn_brain.weights) + ctrnn_brain.input
            )

    for i in range(dim_y):
        for j in range(dim_y):
            states = np.array([Y1[i, j], Y2[i, j]])
            dy_dt = compute_derivatives(states)
            changes_y1[i,j], changes_y2[i,j] = dy_dt

    plt.figure(figsize=(10,6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
    if net_history is not None:
        plt.plot(net_history[:, 0], net_history[:, 1], color='r')
        plt.scatter(net_history[0][0], net_history[0][1], color='orange', zorder=1)
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Phase portrait and a single trajectory for agent brain', fontsize=16)
    plt.show()

def simulate_plot_states(brain, run_duration, input_func, plot=False):
    # simulate_plot_output brain
    step_size = brain.step_size
    # external_inputs = np.zeros(brain.num_neurons) # zero external_inputs
    states_series = []
    # print_list(['{:.2f}'.format(0), brain.get_state()[0], brain.get_state()[1], brain.get_output()[0], brain.get_output()[1]])
    # for _ in range(int(run_duration/step_size)):
    states_series.append(np.copy(brain.get_state()))
    for time in np.arange(step_size, run_duration+step_size, step_size):
        brain.input = input_func(brain.num_neurons)
        brain.euler_step()
        # print_list(['{:.2f}'.format(time), brain.get_state()[0], brain.get_state()[1], brain.get_output()[0], brain.get_output()[1]])
        states_series.append(np.copy(brain.get_state()))
    states_series = np.asarray(states_series)

    # plot oscillator output
    if plot:
        for i in range(brain.num_neurons):
            plt.plot(np.arange(0,run_duration+step_size,step_size),states_series[:,i])
        plt.xlabel('Time')
        plt.ylabel('Neuron states')
        plt.show()
    return states_series

def test_phase_plot():

    brain = BrainCTRNN(
        num_neurons=2,
        step_size=0.01,
        states = np.array([2.5, 5.]),
        # states = np.array([4., 2.]),
        taus=np.array([1.0, 1.0]),
        gains=np.array([1.0, 1.0]),
        biases=np.array([-2.75, -1.75]),
        weights=np.array([4.5, -1, 1, 4.5]), # will be converted to np.array([[4.5, -1], [1, 4.5]])
    )

    brain.compute_output()
    
    input_funct = np.zeros
    states_series = simulate_plot_states(brain, 250, input_funct, plot=True)    
    plot_phase_space(brain, states_series)    


if __name__ == "__main__":
    test_phase_plot()