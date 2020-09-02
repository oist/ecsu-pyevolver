import numpy as np
from ctrnn import BrainCTRNN
import matplotlib.pyplot as plt


def print_list(l):
    print('\t'.join(str(x) for x in l))


def simulate(brain, run_duration):
    # simulate brain
    step_size = brain.step_size
    external_inputs = [0]*brain.num_neurons # zero external_inputs
    outputs = []
    print_list(['{:.2f}'.format(0), brain.get_state()[0], brain.get_state()[1], brain.get_output()[0], brain.get_output()[1]])
    # for _ in range(int(run_duration/step_size)):
    for time in np.arange(step_size, run_duration+step_size, step_size):
        brain.euler_step()
        print_list(['{:.2f}'.format(time), brain.get_state()[0], brain.get_state()[1], brain.get_output()[0], brain.get_output()[1]])
        outputs.append([brain.get_output()[i] for i in range(brain.num_neurons)])
    outputs = np.asarray(outputs)

    # plot oscillator output
    for i in range(brain.num_neurons):
        plt.plot(np.arange(0,run_duration,step_size),outputs[:,i])
    plt.xlabel('Time')
    plt.ylabel('Neuron outputs')
    plt.show()


def run_test_params(run_duration):

    brain = BrainCTRNN(
        num_neurons=2,
        step_size=0.01,
        states = np.array([0.5, 0.5]),
        taus=np.array([1.0, 1.0]),
        gains=np.array([1.0, 1.0]),
        biases=np.array([-2.75, -1.75]),
        weights=np.array([4.5, -1, 1, 4.5]), # will be converted to np.array([[4.5, -1], [1, 4.5]])
    )

    brain.compute_output()

    simulate(brain, run_duration)

    last_output = [brain.get_output()[i] for i in range(brain.num_neurons)]
    assert last_output == [0.810528526040536, 0.5314836931489646]


def run_test_random(num_neurons, run_duration):
    # init params
    step_size = 0.01

    brain = BrainCTRNN(
        num_neurons=num_neurons,
        step_size=step_size,
        tau_range=(1, 10),
        gain_range=(1, 1),
        bias_range=(-15, 15),
        weight_range=(-15, 15),
        state_range=(0, 0.5)
    )

    brain.compute_output()
    simulate(brain, run_duration)


if __name__ == "__main__":
    run_test_params(250)
    # run_test_random(2,250)
    # run_test_random(5, 250)
