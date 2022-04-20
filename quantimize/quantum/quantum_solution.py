import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fractions import Fraction

from qiskit import IBMQ
from qiskit_optimization import QuadraticProgram
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA


def sample_grid():
    """
    This function generates the cost grid for a scaled-down experiment example (4 qubits)

    :return: the cost grid
    """
    cg = -1 * np.array([[5, 5, 5, 5], [5, 1, 5, 5], [5, 1, 5, 5], [5, 5, 5, 5]])
    # Without the coefficient -1, we would obtain a maximization problem.

    # The cost grid - the value represents the cost to travel through that grid, with dimension N x N
    # The cost should be computed mainly based on atmospheric data, plus a penalty by deviating from the 
    # straight-line solution. The idea behind is that a grid too far away from the straight-line solution should be
    # costly even if the non-CO2 emission there is low, because the CO2 emission, which depends solely on distance of 
    # travel, will be high.
    return cg


def sample_cost(z):
    """
    Evaluates the cost function of the example problem, given the Pauli-z values of the 4 qubits.
    :param z: the Pauli-z values of the 4 qubits which is an element of the set {-1,1}^4
    :return: the value of the cost function
    """
    z1, z2, z3, z4 = z
    return -1 * (3*z1*z2 + z1*z3 + 5*z2*z4 + 3*z3*z4 + 6*z1 - 10*z4)


def brute_force():
    """
    Solve by brute force the sample problem
    :return: tuple containing the optimal solution and the optimal cost
    """
    sol_list = []
    cost_list = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                for l in [-1, 1]:
                    sol = [i, j, k, l]
                    cost = sample_cost([i, j, k, l])
                    print('z =', sol, 'gives cost:', cost)
                    sol_list.append(sol)
                    cost_list.append(cost)
    return sol_list[np.argmin(cost_list)], np.min(cost_list)


def obtain_weight_matrices(cg):
    """
    obtain the interaction strengths (edges) for the qubit grid from the cost grid
    :param cg: cost grid
    :return: interaction strengths (edges) stored in two separate matrices, one for vertical, one for horizontal
    """
    wh = np.array([[(cg[i][j+1]+cg[i][j])/2 for j in range(len(cg)-1)] for i in range(len(cg))])
    # The matrix storing horizontal edge (coupling) values between voxel centers (qubits)
    # with dimension N x N-1
    wv = np.array([[(cg[i][j]+cg[i+1][j])/2 for j in range(len(cg))] for i in range(len(cg)-1)])
    # The matrix storing vertical edge (coupling) values between voxel centers (qubits)
    # with dimension N-1 x N
    return wv, wh

def create_vcg(n, orientation=0):  #voxel_center_graph
    """
    create the initial voxel_center_graph
    :param n:
    :param orientation:
    :return:
    """
    vcg = np.zeros((n,n))
    if orientation == 0:  # travel from southwest to northeast or the opposite
        for i in range(n):
            for j in range(n):
                if (i == 0 and j != n-1) or (j == 0 and i != n-1):
                    vcg[i][j] = 1
                if (i == n-1 and j != 0) or (j == n-1 and i != 0):
                    vcg[i][j] = -1
    if orientation == 1:  # travel from northwest to southeast or the opposite
        for i in range(n):
            for j in range(n):
                if (i == 0 and j != 0) or (j == n-1 and i != n-1):
                    vcg[i][j] = 1
                if (i == n-1 and j != n-1) or (j == 0 and i != 0):
                    vcg[i][j] = -1
    return vcg

def construct_function(cg, orientation=0):
    """Constructs the mathematical function to be optimized

    Args:
        cg: cost grid

    Returns:
        sympyfunction, dict: mathematical sympy function with coefficients
    """
    function = 0
    sp.init_printing(use_unicode=True)
    
    n = len(cg)
    
    vcg = create_vcg(n, orientation=orientation)
    
    wv, wh = obtain_weight_matrices(cg)
    
    # construct the Pauli operators Zij 
    for i in range(1,n-1):
        for j in range(1, n-1):
            globals()['Z%s %x' % (i, j)] = sp.symbols('Z'+str(i)+str(j))
        
    # for each qubit, add its contribution to the total cost function
    for i in range(1,n-1):
        for j in range(1, n-1):
            term = 0
            if vcg[i-1][j] == 0:  # Check if neighbor up is variable or fixed
                term += globals()['Z%s %x' % (i-1, j)] * globals()['Z%s %x' % (i, j)] * wv[i-1][j] / 2 
                # divide by 2 because each edge joining two variable qubits will be counted twice
            else:
                term += vcg[i-1][j] * globals()['Z%s %x' % (i, j)] * wv[i-1][j]
            
            if vcg[i+1][j] == 0:  # Check if neighbor down is variable or fixed
                term += globals()['Z%s %x' % (i+1, j)] * globals()['Z%s %x' % (i, j)]  * wv[i][j] / 2 
                # divide by 2 because each edge joining two variable qubits will be counted twice
            else:
                term += vcg[i+1][j] * globals()['Z%s %x' % (i, j)] * wv[i][j]

            if vcg[i][j-1] == 0:  # Check if neighbor left is variable or fixed
                term += globals()['Z%s %x' % (i, j-1)] * globals()['Z%s %x' % (i, j)]  * wh[i][j-1] / 2 
                # divide by 2 because each edge joining two variable qubits will be counted twice
            else:
                term += vcg[i][j-1] * globals()['Z%s %x' % (i, j)] * wh[i][j-1]
                
            if vcg[i][j+1] == 0:  # Check if neighbor right is variable or fixed
                term += globals()['Z%s %x' % (i, j+1)] * globals()['Z%s %x' % (i, j)]  * wh[i][j] / 2 
                # divide by 2 because each edge joining two variable qubits will be counted twice
            else:
                term += vcg[i][j+1] * globals()['Z%s %x' % (i, j)] * wh[i][j]
            
            function += term
                
    print('Cost function in Ising Hamiltonian form:', function)
    
    # map the Pauli Z variable {-1, 1} to variable x {0, 1} by doing Z = 2x-1
    for i in range(1,n-1):
        for j in range(1, n-1):
            function = function.subs(sp.symbols('Z'+str(i)+str(j)), 2*sp.symbols('q'+str(i)+str(j)) -1 )
        coeffs = sp.Poly(function).as_dict()
        
        # expand and simplify the cost function
    function = sp.expand(function)
    
    print('Cost function in QUBO form:', function)
        
    # function in sympy form and coefficients of all terms in an easily readble dictionary form returned
    return function, coeffs

def generate_QP(coeffs, n, verbose=False):
    """Generates the Quadratic Program to be optimized

    Args:
        coeffs (dict): Contains the coefficients from the sympy funciton
        n (int): Number of hydraulic heads

    Returns:
        QuadraticProgram: The quadratic program to optimize
    """
    qp = QuadraticProgram()   #Initialize a Qiskit Quadratic Program object
    for i in range(1,n-1):
        for j in range(1,n-1):
            qp.binary_var('q'+str(i)+str(j))  # binary variables xij
    constant = 0      # constant term
    linear = {}      # coefficients for linear terms
    quadratic = {}    # coefficients for coupling terms
    
    nl = [(i,j) for i in range(1,n-1) for j in range(1,n-1)]  # name list
    for key,value in coeffs.items():    # add coefficients to the corresponding dictionaries one by one
        if sum(key) == 0:
            constant = float(value)
        elif sum(key) == 1:
            term = 'q'+str(nl[np.argmax(key)][0])+str(nl[np.argmax(key)][1])
            linear[term] = float(value)
        else:
            indices = [i[0] for i in np.argwhere(np.array(key)>0)]
            term = ('q'+str(nl[indices[0]][0])+str(nl[indices[0]][1]), 'q'+str(nl[indices[1]][0])+str(nl[indices[1]][1]))
            quadratic[term] = float(value)
    qp.minimize(linear=linear, quadratic=quadratic, constant=constant)  # run the optimization algorithm, find minimum value
    if verbose:
        print(qp.export_as_lp_string())  # make a printed report of the task
    return qp

def run_QAOA(cg, orientation=0, verbose=False):
    """Constructs and solves the mathematical function

    Args:
        cg: cost grid
        verbose (Bool): Output information or not

    Returns:
        
    """
    n = len(cg)
    # Create the mathematical function
    function, coeffs = construct_function(cg, orientation=orientation)
    if verbose:
        print('Function in Sympy:', function)
    # Generate the quadratic problem and solve it with Qiskit
    qp = generate_QP(coeffs,n,verbose)
    qins = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=1000, seed_simulator=123)
    meo = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=1, quantum_instance=qins))  # solve with QAOA algorithm,
    result = meo.solve(qp)    #could replace by classical solver and other quantum solvers
    z_sol = 2*result.x-1
    q_sol = np.array([int((i+1)%2) for i in result.x])
    if verbose:
        print('\nrun time:', result.min_eigen_solver_result.optimizer_time)
        print('result in Pauli-Z form:', z_sol)
        print('result in qubit form:', q_sol)
    vcg = create_vcg(n, orientation)
    nl = [(i,j) for i in range(1,n-1) for j in range(1,n-1)]  # name list
    for l in range(len(z_sol)):
        vcg[nl[l][0]][nl[l][1]] = z_sol[l]
    return z_sol, q_sol, vcg