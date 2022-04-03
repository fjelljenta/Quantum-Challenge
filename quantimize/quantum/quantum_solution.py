import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from noisyopt import minimizeSPSA
from quantimize.quantum.toolbox import tensorize_flight_info, normalize_input_data, sigmoid
from quantimize.classic.toolbox import curve_3D_solution, compute_cost
import quantimize.data_access as da

# Adjustable parameters include the ansatz, the layer of ansatz, the embedding method, dxy, dz, c,
# indices in distribution, minimzer (could go for PyTorch or gradient methods or adjust params in SPSA), etc.


def quantum_neural_network(flight_nr, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    data = normalize_input_data(tensorize_flight_info())[flight_nr]

    @qml.qnode(dev)
    def circuit(params):
        qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), normalize=True, pad_with=0.)
        qml.StronglyEntanglingLayers(params, wires=list(range(n_qubits)))
        #return qml.expval(qml.PauliZ(0))
        return qml.probs(wires=range(n_qubits))

    num_layers = 5

    flat_shape = num_layers * n_qubits * 3
    param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=n_qubits, n_layers=num_layers)
    init_params = np.random.normal(scale=0.1, size=param_shape)

    init_params_spsa = init_params.reshape(flat_shape)

    plot = plt.bar(np.arange(2**n_qubits), circuit(init_params))
    plt.show()

    qnode = qml.QNode(circuit, dev)

    def from_distribution_to_trajectory(distribution):
        c = 2 ** (n_qubits-2)
        dxy = 0.01
        dz = 100
        index_xy = np.sort(np.argsort(distribution[int(0.5*2**(n_qubits-1)):2**(n_qubits-1)])[::-1][:3])
        value_xy = distribution[index_xy]
        normalized_xy = sigmoid(value_xy - np.mean(value_xy), c) - 0.5


        index_z = np.sort(np.argsort(distribution[2**(n_qubits-1):int(1.5*2**(n_qubits-1))])[::-1][:5])
        value_z = distribution[index_z+2**(n_qubits-1)]
        normalized_z = sigmoid(value_z - np.mean(value_z), c) - 0.5

        info = da.get_flight_info(flight_nr)
        slope = (info['end_latitudinal'] - info['start_latitudinal']) * 111 / \
                ((info['end_longitudinal'] - info['start_longitudinal']) * 85)
        flight_level = info['start_flightlevel']

        perp_slope = -1/slope
        ctrl_pts_xy = [], []
        for i in range(3):
            index = index_xy[i]
            size = normalized_xy[i]
            intersection = index/(2**(n_qubits-1)) * info['end_longitudinal'] + \
                           (1-index/(2**(n_qubits-1))) * info['start_longitudinal'], \
                           index/(2 ** (n_qubits - 1)) * info['end_latitudinal'] + \
                           (1 - index / (2 ** (n_qubits - 1))) * info['start_latitudinal']
            ctrl_pts_xy[0].append(np.cos(np.arctan(perp_slope)) * dxy * size + intersection[0])
            ctrl_pts_xy[1].append(np.sin(np.arctan(perp_slope)) * dxy * size + intersection[1])

        ctrl_pts_z = []
        for i in range(5):
            #index = index_z[i]
            size = normalized_z[i]
            ctrl_pts_z.append(size*dz + flight_level)

        ctrl_pts = ctrl_pts_xy[0] + ctrl_pts_xy[1] + ctrl_pts_z
        print(ctrl_pts)
        trajectory = curve_3D_solution(flight_nr, ctrl_pts)
        return trajectory

    def cost_spsa(params):
        distribution = np.array(qnode(params.reshape(num_layers, n_qubits, 3)))
        trajectory = from_distribution_to_trajectory(distribution)
        cost = compute_cost(trajectory)
        return cost

    #return cost_spsa(init_params)

    niter_spsa = 20

    # Evaluate the initial cost
    cost_store_spsa = [cost_spsa(init_params)]
    device_execs_spsa = [0]

    def callback_fn(xk):
        cost_val = cost_spsa(xk)
        cost_store_spsa.append(cost_val)

        # We've evaluated the cost function, let's make up for that
        num_executions = int(dev.num_executions / 2)
        device_execs_spsa.append(num_executions)

        iteration_num = len(cost_store_spsa)
        if iteration_num % 1 == 0:
            print(
                f"Iteration = {iteration_num}, "
                f"Number of device executions = {num_executions}, "
                f"Cost = {cost_val}"
            )

    res = minimizeSPSA(
        cost_spsa,
        x0=init_params_spsa.copy(),
        niter=niter_spsa,
        paired=False,
        c=0.15,
        a=0.2,
        callback=callback_fn,
    )

    return res