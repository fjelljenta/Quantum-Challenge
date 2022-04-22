# Quantum-Challenge by Deloitte
Hello everybody,
this is team Quantimize's contribution to the Quantum Climate Challenge by Deloitte (https://app.ekipa.de/challenges/deloitte-quantum/brief).
In this repository you can find all our code we produced during the challenge.

You can find a file called requirement.txt, which provides all the packages necessary to run our code. 

The folder Quantimze contains all the functions we have written. It is devided into parts, which are relevant only for our classical solution, only for our quantum solutions or for both, like the data extraction and access or the safety check. The data used were provided by Deloitte.

In our work we implemented the nowadays fuel-efficient straight line solution and a classical genetic algorithm. The implementation of the genetic algorithm was inspired by the paper "Air traffic simulation in chemistry-climate model EMAC 2.41: AirTraf 1.0" written by Hiroshi Yamashita et al. It is used to benchmark our quantum algorithms.

Moreover, we discussed three types of quantum solutions:  a Quantum Approximate Optimization Algorithm (QAOA) for Quadratic Unconstrained Binary Optimization (QUBO) problem, the quantum genetic algorithm and a quantum neural network. 

We found out, that at the moment the classical genetic algorithm is the best choice for a big data set.  However, we are confident, that this will change with larger and better Noisy Intermediate-Scale Quantum (NISQ) hardware. The most promising quantum algorithm we found is to discretize the space,transform the cost function into QUBO form and then solve it by quantum annealing or the QAOA algorithm on gate-based quantum computers such as IBM’s superconducting quantum devices. A four-qubit device is enough to build a scaled-down problem for demonstration purpose. However, more qubits are needed to run it on the big data set to avoid oversimplification.

The quantum genetic algorithm performs solid, but does not give an obvious advantage over the classical genetic algorithm.The quantum neural network (QNN) we implemented, is not trainable with the currently available number of qubits. However, we do believe that this is a promising concept, which can be useful in the future, when larger devicesare standar

The Quantimize.ipynb summarizes everything we have done to give you a brief overview. 

If you like to test our code in more depth. We recommend the notebook „Tutorial.ipynb“, which explains how to call and use our functions in more detail.
