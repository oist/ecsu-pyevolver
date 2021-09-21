import numpy as np
from pyevolver.ctrnn import BrainCTRNN
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_ctrnn_and_get_states(brain, run_duration, input_func=None, plot=False):
    # simulate_plot_output brain
    step_size = brain.step_size
    num_steps = int(run_duration/step_size)
    num_states = len(brain.states)
    states_series = np.zeros((num_steps, num_states))
    state_derivative_series = np.zeros((num_steps, num_states))
    time_series = np.arange(step_size, run_duration+step_size, step_size)
    for step in range(num_steps):
        states_series[step] = brain.get_state()        
        if input_func is not None:
            brain.input = input_func(brain.num_neurons)
        brain.euler_step()
        state_derivative_series[step] = brain.dy_dt
        step += 1        

    # plot output
    if plot:
        for i in range(brain.num_neurons):
            plt.plot(np.arange(0,run_duration+step_size,step_size),states_series[:,i])
        plt.xlabel('Time')
        plt.ylabel('Neuron states')
        plt.show()
    return time_series, states_series, state_derivative_series


def plot_phase_space_traj():
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
    ymin = -5
    ymax = 5
    num_steps = 10
    y1 = np.linspace(ymin, ymax, num_steps)
    y2 = np.linspace(ymin, ymax, num_steps)

    plt.figure(figsize=(10,6))

    pbar = tqdm(total=num_steps**2)
    input_func = np.zeros

    for i in y1:
        for j in y2:
            # print(i,j)
            pbar.update()
            brain = BrainCTRNN(
                num_neurons=2,
                step_size=0.01,
                states = np.array([i, j]),
                taus=np.array([1.0, 1.0]),
                gains=np.array([1.0, 1.0]),
                biases=np.array([-2.75, -1.75]),
                weights=np.array([4.5, -1, 1, 4.5]) # will be converted to np.array([[4.5, -1], [1, 4.5]])
            )
            # brain.input = np.array([1.,1.])
            brain.compute_output()
            # input_func = np.zeros            
            _, states_series, _ = run_ctrnn_and_get_states(brain, 100, input_func)
            plt.plot(states_series[:, 0], states_series[:, 1], color='r')
            plt.scatter(states_series[0][0], states_series[0][1], color='orange', zorder=1)
    
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Multi-trajectory for agent brain', fontsize=16)
    plt.show()


def test_control_phase_plot():
    from control.phaseplot import phase_plot

    state_ctrnn_brain = {} # state_tuple -> state_derivative_series of ctrnn_brain
    input_func = np.zeros

    step_size = 0.1
    run_duration = 1
    num_steps = int(run_duration/step_size)

    time_series = np.arange(0, run_duration, step_size)

    xmin = -5
    xmax = 5
    xsteps = 10

    x0 = np.linspace(xmin, xmax, xsteps)
    x1 = np.linspace(xmin, xmax, xsteps)

    def get_ctrnn_brain_dydt(state_tuple, t):
        if type(state_tuple) == np.ndarray:
            state_tuple = tuple(state_tuple.tolist())
        if state_tuple in state_ctrnn_brain:
            result =  state_ctrnn_brain[state_tuple]
        else:
            ctrnn = \
                BrainCTRNN(
                    num_neurons=2,
                    step_size=step_size,
                    states = np.array(state_tuple),
                    taus=np.array([1.0, 1.0]),
                    gains=np.array([1.0, 1.0]),
                    biases=np.array([-2.75, -1.75]),
                    weights=np.array([4.5, -1, 1, 4.5]), # will be converted to np.array([[4.5, -1], [1, 4.5]])
                )
            ctrnn.compute_output()
            _, _, state_derivative_series = run_ctrnn_and_get_states(ctrnn, run_duration, input_func)
            result = state_ctrnn_brain[state_tuple] = state_derivative_series
        idx = (np.abs(time_series - t)).argmin()
        return result[idx]

    phase_plot(
        get_ctrnn_brain_dydt, 
        [-5, 5, 10], 
        [-5, 5, 10], 
        # X0=[(i,j) for i in x0 for j in x1],
        # scale=5,
        T=time_series,
        # timepts=[0,1]
    )
    plt.show()


if __name__ == "__main__":
    # run_test_params(250)
    # run_test_random(2,250)
    # run_test_random(5, 250)
    # plot_phase_space_traj()
    test_control_phase_plot()
