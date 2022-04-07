#########################################################
#        Improved and adjusted version of the           #
#       QUANTUM GENETIC ALGORITHM (24.05.2016)          #
#                                                       #
#               by R. Lahoz-Beltra                      #
#                                                       #
#                                                       #
#########################################################
import math
import numpy as np
from quantimize.classic.toolbox import curve_3D_trajectory_core, compute_cost
import quantimize.data_access as da
import quantimize.converter as cv
from quantimize.visualisation import *

def curve(X, Y):
    def f(x):
        for i in range(len(X)-1):
            if x>=X[i] and x<=X[i+1]:
                return Y[i]+(x-X[i])*(Y[i+1]-Y[i])/(X[i+1]-X[i])
    return f

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
N=10                  # Define here the population size
Genome=50              # Define here the chromosome length
generation_max=2000   # Define here the maximum number of
                      # generations/iterations

#########################################################
# VARIABLES ALGORITHM                                   #
#########################################################
popSize=N+1
genomeLength=Genome+1
top_bottom=3
QuBitZero = np.array([[1],[0]])
QuBitOne = np.array([[0],[1]])
AlphaBeta = np.empty([top_bottom])
fitness = np.empty([popSize])
probability = np.empty([popSize])
# qpv: quantum chromosome (or population vector, QPV)
qpv = np.empty([popSize, genomeLength, top_bottom])
nqpv = np.empty([popSize, genomeLength, top_bottom])
# chromosome: classical chromosome
chromosome = np.empty([popSize, genomeLength],dtype=np.int)
child1 = np.empty([popSize, genomeLength, top_bottom])
child2 = np.empty([popSize, genomeLength, top_bottom])
best_chrom = np.empty([generation_max])

# Initialization global variables
theta=0;
iteration=0;
the_best_chrom=0;
generation=0;

#########################################################
# QUANTUM POPULATION INITIALIZATION                     #
#########################################################
def Init_population():
    # Hadamard gate
    r2=math.sqrt(2.0)
    h=np.array([[1/r2,1/r2],[1/r2,-1/r2]])
    # Rotation Q-gate
    theta=0;
    rot =np.empty([2,2])
    # Initial population array (individual x chromosome)
    i=1; j=1;
    for i in range(1,popSize):
     for j in range(1,genomeLength):
        theta=np.random.uniform(0,1)*90
        theta=math.radians(theta)
        rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta);
        rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta);
        AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0])+rot[0,1]*(h[0][1]*QuBitZero[1])
        AlphaBeta[1]=rot[1,0]*(h[1][0]*QuBitZero[0])+rot[1,1]*(h[1][1]*QuBitZero[1])
        # alpha squared
        qpv[i,j,0]=np.around(2*pow(AlphaBeta[0],2),2)
        # beta squared
        qpv[i,j,1]=np.around(2*pow(AlphaBeta[1],2),2)



#########################################################
# MAKE A MEASURE                                        #
#########################################################
# p_alpha: probability of finding qubit in alpha state
def Measure(p_alpha):
    for i in range(1,popSize):
        print()
        for j in range(1,genomeLength):
            if p_alpha<=qpv[i, j, 0]:
                chromosome[i,j]=0
            else:
                chromosome[i,j]=1
            print(chromosome[i,j]," ",end="")
        print()
    print()

#########################################################
# FITNESS EVALUATION                                    #
#########################################################
def Fitness_evaluation(flight_nr, generation):
    i=1; j=1; fitness_total=0; sum_sqr=0;
    fitness_average=0; variance=0;
    info = da.get_flight_info(flight_nr)
    for i in range(1,popSize):
        fitness[i]=0

    for i in range(1,popSize):
        x=[0,0,0]
        for j in range(5):
            # translate from binary to decimal value
            x[0]=x[0]+chromosome[i,j+1]*pow(2,5-j-1)
            x[1]=x[1]+chromosome[i,j+6]*pow(2,5-j-1)
            x[2]=x[2]+chromosome[i,j+11]*pow(2,5-j-1)
        y=[0,0,0]
        for j in range(4):
            y[0]=y[0]+chromosome[i,j+16]*pow(2,4-j-1)
            y[1]=y[1]+chromosome[i,j+20]*pow(2,4-j-1)
            y[2]=y[2]+chromosome[i,j+24]*pow(2,4-j-1)
        z=[0,0,0,0,0]
        for j in range(4):
            z[0]+=chromosome[i,j+31]*pow(2,4-j-1)
            z[1]+=chromosome[i,j+35]*pow(2,4-j-1)
            z[2]+=chromosome[i,j+39]*pow(2,4-j-1)
            z[3]+=chromosome[i,j+43]*pow(2,4-j-1)
            z[4]+=chromosome[i,j+47]*pow(2,4-j-1)

        for j in range(3):
            x[j]=x[j]*(abs(info['start_longitudinal']-info['end_longitudinal']))/31 - min(info['start_longitudinal'],info['end_longitudinal'])
            y[j]=y[j]*26/15 + 34

        for j in range(5):
            z[j]=z[j]*300/15 + 100

        s=np.argsort(x)
        if info['start_longitudinal']>info['end_longitudinal']:
            s=s[::-1]
        x=[x[index] for index in s]
        x=[info['start_longitudinal']]+x+[info['end_longitudinal']]
        y=[y[index] for index in s]
        y=[info['start_latitudinal']]+y+[info['end_latitudinal']]
        z=[info['start_flightlevel']] + z
        print("ctrl_pts: ", x, y, z)
        # ctrl_pts=xy_sorted+z # ctrl_points we need
        total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                    info['end_longitudinal'], info['end_latitudinal'])
        curve_xy=curve(x, y)
        curve_z=curve(np.linspace(0, total_distance, 6), z)
        trajectory=curve_3D_trajectory_core(flight_nr, curve_xy, curve_z, 0.3)
        fitness[i]= -compute_cost({"trajectory": trajectory}) # - because we want t minimalize the cost

        #########################################################

        print("fitness = ",i," ",fitness[i])
        fitness_total=fitness_total+fitness[i]

    fitness_average=fitness_total/N
    i=1;
    while i<=N:
        sum_sqr=sum_sqr+pow(fitness[i]-fitness_average,2)
        i=i+1
    variance=sum_sqr/N
    if variance<=1.0e-4:
        variance=0.0
    # Best chromosome selection
    the_best_chrom=0;
    fitness_max=fitness[1];
    for i in range(1,popSize):
        if fitness[i]>=fitness_max:
            fitness_max=fitness[i]
            the_best_chrom=i
    best_chrom[generation]=the_best_chrom

    print("Population size = ",popSize - 1)
    print("mean fitness = ",fitness_average)
    print("variance = ",variance," Std. deviation = ",math.sqrt(variance))
    print("fitness max = ",best_chrom[generation])
    print("Chromosome with max fitness", chromosome[int(best_chrom[generation])])
    print("fitness sum = ",fitness_total)

    if generation==generation_max-1:

        i=int(best_chrom[generation])

        x=[0,0,0]
        for j in range(5):
            # translate from binary to decimal value
            x[0]=x[0]+chromosome[i,j+1]*pow(2,5-j-1)
            x[1]=x[1]+chromosome[i,j+6]*pow(2,5-j-1)
            x[2]=x[2]+chromosome[i,j+11]*pow(2,5-j-1)
        y=[0,0,0]
        for j in range(4):
            y[0]=y[0]+chromosome[i,j+16]*pow(2,4-j-1)
            y[1]=y[1]+chromosome[i,j+20]*pow(2,4-j-1)
            y[2]=y[2]+chromosome[i,j+24]*pow(2,4-j-1)
        z=[0,0,0,0,0]
        for j in range(4):
            z[0]+=chromosome[i,j+31]*pow(2,4-j-1)
            z[1]+=chromosome[i,j+35]*pow(2,4-j-1)
            z[2]+=chromosome[i,j+39]*pow(2,4-j-1)
            z[3]+=chromosome[i,j+43]*pow(2,4-j-1)
            z[4]+=chromosome[i,j+47]*pow(2,4-j-1)

        for j in range(3):
            x[j]=x[j]*60/31 - 30
            y[j]=y[j]*26/15 + 34

        for j in range(5):
            z[j]=z[j]*300/15 + 100

        s=np.argsort(x)
        if info['start_longitudinal']>info['end_longitudinal']:
            s=s[::-1]
        x=[x[index] for index in s]
        x=[info['start_longitudinal']]+x+[info['end_longitudinal']]
        y=[y[index] for index in s]
        y=[info['start_latitudinal']]+y+[info['end_latitudinal']]
        z=[info['start_flightlevel']] + z
        print("ctrl_pts: ", x, y, z)
        # ctrl_pts=xy_sorted+z # ctrl_points we need
        total_distance = cv.coordinates_to_distance(info['start_longitudinal'], info['start_latitudinal'],
                                                    info['end_longitudinal'], info['end_latitudinal'])
        curve_xy=curve(x, y)
        curve_z=curve(np.linspace(0, total_distance, 6), z)
        trajectory=curve_3D_trajectory_core(flight_nr, curve_xy, curve_z, 0.3)
        return trajectory

#########################################################
# QUANTUM ROTATION GATE                                 #
#########################################################
def rotation():
    rot=np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(1,popSize):
       for j in range(1,genomeLength):
           best_chrom[generation]=int(best_chrom[generation])
           if fitness[i]<fitness[int(best_chrom[generation])]:
             # if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:
               if chromosome[i,j]==0 and chromosome[int(best_chrom[generation]),j]==1:
                   # Define the rotation angle: delta_theta (e.g. 0.0785398163)
                   delta_theta=0.0785398163
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
               if chromosome[i,j]==1 and chromosome[int(best_chrom[generation]),j]==0:
                   # Define the rotation angle: delta_theta (e.g. -0.0785398163)
                   delta_theta=-0.0785398163
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
             # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

#########################################################
# X-PAULI QUANTUM MUTATION GATE                         #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit
def mutation(pop_mutation_rate, mutation_rate):

    for i in range(1,popSize):
        up=np.random.random_integers(100)
        up=up/100
        if up<=pop_mutation_rate:
            for j in range(1,genomeLength):
                um=np.random.random_integers(100)
                um=um/100
                if um<=mutation_rate:
                    nqpv[i,j,0]=qpv[i,j,1]
                    nqpv[i,j,1]=qpv[i,j,0]
                else:
                    nqpv[i,j,0]=qpv[i,j,0]
                    nqpv[i,j,1]=qpv[i,j,1]
        else:
            for j in range(1,genomeLength):
                nqpv[i,j,0]=qpv[i,j,0]
                nqpv[i,j,1]=qpv[i,j,1]
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            qpv[i,j,0]=nqpv[i,j,0]
            qpv[i,j,1]=nqpv[i,j,1]



########################################################
#                                                      #
# MAIN PROGRAM                                         #
#                                                      #
########################################################
def Q_GA(flight_nr):
    generation=0;
    Init_population()
    Measure(0.5)
    Fitness_evaluation(flight_nr, generation)
    while (generation<generation_max-2):
      rotation()
      mutation(0.01,0.001)
      generation=generation+1
      Measure(0.5)
      Fitness_evaluation(flight_nr, generation)
    rotation()
    mutation(0.01,0.001)
    generation=generation+1
    Measure(0.5)
    trajectory=Fitness_evaluation(flight_nr, generation)
    return trajectory
