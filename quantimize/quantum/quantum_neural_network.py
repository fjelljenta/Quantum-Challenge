import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from noisyopt import minimizeSPSA
import quantimize.quantum.toolbox as qtoolbox
import quantimize.classic.toolbox as ctoolbox
import quantimize.data_access as da

# Adjustable parameters include the ansatz, the layer of ansatz, the embedding method, dxy, dz, c,
# indices in distribution, minimzer (could go for PyTorch or gradient methods or adjust params in SPSA), etc.


def quantum_neural_network(flight_nr, n_qubits, init_solution):
    """Create a quantum neural network for a given flight with n_qubits and an initial solution.

    Args:
        flight_nr (int): flight number
        n_qubits (int): number of qubits the QNN should operate on
        init_solution (list): list of initial boundary points

    Returns:
        scipy.optimize: result of the optimization containing the boundaries and the cost
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    data = qtoolbox.normalize_input_data(
        qtoolbox.tensorize_flight_info())[flight_nr]

    @qml.qnode(dev)
    def circuit(params):
        """Create a circuit with params

        Args:
            params (list): list of parameters

        Returns:
            quantum circuit: returns the basic quantum circuit
        """
        #qml.IQPEmbedding(data, wires=range(n_qubits))
        #qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), normalize=True, pad_with=0.)
        qml.StronglyEntanglingLayers(params, wires=list(range(n_qubits)))
        # return qml.expval(qml.PauliZ(0))
        return qml.probs(wires=range(n_qubits))

    num_layers = 5

    flat_shape = num_layers * n_qubits * 3
    param_shape = qml.templates.StronglyEntanglingLayers.shape(
        n_wires=n_qubits, n_layers=num_layers)
    init_params = np.random.normal(scale=0.1, size=param_shape)

    init_params_spsa = init_params.reshape(flat_shape)

    #plot = plt.bar(np.arange(2**n_qubits), circuit(init_params))
    # plt.show()

    qnode = qml.QNode(circuit, dev)

    def from_distribution_to_trajectory_2(distribution):
        """Create a trajectory from a given distribution

        Args:
            distribution (list): list with points for the distribution

        Returns:
            list: list of trajectory points
        """
        off_setted_distribution = distribution - 1 / 2 ** n_qubits
        tabx = np.linspace(0, 1, 2 ** n_qubits)

        # Generating weights for polynomial function with degree =2
        weights = np.polyfit(tabx, off_setted_distribution, 2)

        # Generating model with the given weights
        model = np.poly1d(weights)

        # Prediction on validation set
        # We will plot the graph for 70 observations only

        #plt.scatter(tabx, off_setted_distribution, facecolor='None', edgecolor='k', alpha=0.3)
        #plt.plot(tabx, model(tabx))
        # plt.show()

        ctrl_pts = np.concatenate((np.array(init_solution[:6]),
                                  np.array(init_solution[6:] + 1e4 * model(np.linspace(0, 1, 5)))))
        #trajectory = curve_3D_solution(flight_nr, ctrl_pts)
        trajectory = ctoolbox.curve_3D_trajectory(flight_nr, ctrl_pts)
        corrected_trajectory = ctoolbox.correct_for_boundaries(trajectory)
        return corrected_trajectory

    def from_distribution_to_trajectory(distribution):
        """Create a trajectory from a given distribution

        Args:
            distribution (list): list with points for the distribution

        Returns:
            list: list of trajectory points
        """
        c = 2 ** (n_qubits-2)
        dxy = 0.01
        dz = 100
        index_xy = np.sort(np.argsort(
            distribution[int(0.5*2**(n_qubits-1)):2**(n_qubits-1)])[::-1][:3])
        value_xy = distribution[index_xy]
        normalized_xy = qtoolbox.sigmoid(value_xy - np.mean(value_xy), c) - 0.5

        index_z = np.sort(np.argsort(
            distribution[2**(n_qubits-1):int(1.5*2**(n_qubits-1))])[::-1][:5])
        value_z = distribution[index_z+2**(n_qubits-1)]
        normalized_z = qtoolbox.sigmoid(value_z - np.mean(value_z), c) - 0.5

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
                (1 - index / (2 ** (n_qubits - 1))) * \
                info['start_latitudinal']
            ctrl_pts_xy[0].append(
                np.cos(np.arctan(perp_slope)) * dxy * size + intersection[0])
            ctrl_pts_xy[1].append(
                np.sin(np.arctan(perp_slope)) * dxy * size + intersection[1])

        ctrl_pts_z = []
        for i in range(5):
            #index = index_z[i]
            size = normalized_z[i]
            ctrl_pts_z.append(size*dz + flight_level)

        ctrl_pts = ctrl_pts_xy[0] + ctrl_pts_xy[1] + ctrl_pts_z
        #trajectory = curve_3D_solution(flight_nr, ctrl_pts)
        trajectory = ctoolbox.curve_3D_trajectory(flight_nr, ctrl_pts)
        corrected_trajectory = ctoolbox.correct_for_boundaries(trajectory)
        return corrected_trajectory

    def cost_spsa(params):
        """Calculates the cost for a certain distrubution

        Args:
            params (list): list of parameters

        Returns:
            float: cost of the distribution
        """
        distribution = np.array(qnode(params.reshape(num_layers, n_qubits, 3)))
        trajectory = ctoolbox.correct_for_boundaries(
            from_distribution_to_trajectory_2(distribution))
        #trajectory = {'flight_nr':flight_nr, 'trajectory':trajectory}
        cost = ctoolbox.compute_cost(trajectory)
        return cost

    # return cost_spsa(init_params)

    niter_spsa = 50

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
        if iteration_num % 2 == 0:
            print(
                f"Iteration = {iteration_num}, "
                f"Number of device executions = {num_executions}, "
                f"Cost = {cost_val}"
            )
    # run the optimization
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
