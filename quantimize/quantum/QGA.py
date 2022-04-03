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
from quantimize.classic.toolbox import curve_3D_trajectory, compute_cost
import quantimize.data_access as da

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
N=10                  # Define here the population size
Genome=4              # Define here the chromosome length
generation_max=100    # Define here the maximum number of 
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
    for i in range(1,popSize):
        fitness[i]=0

    for i in range(1,popSize):
      x1=0;
       for j in range(1,genomeLength):
           # translate from binary to decimal value
           x1=x1+chromosome[i,j]*pow(2,genomeLength-j-1)
      x2=0
      for j in range(1, genomeLength-1):
        x2=x2+chromosome[i,j+genomeLength]*pow(2,genomeLength-1-j-1)
      x3=0
      for j in range(1, genomeLength-2):
        x3=x3+chromosome[i,j+genomeLength]*pow(2,genomeLength-2-j-1)
      y1=0;
       for j in range(1,genomeLength):
           y1=y1+chromosome[i,j]*pow(2,genomeLength-j-1)
      y2=0
      for j in range(1, genomeLength-1):
        y2=y2+chromosome[i,j+genomeLength]*pow(2,genomeLength-1-j-1)
      y3=0
      for j in range(1, genomeLength-2):
        y3=y3+chromosome[i,j+genomeLength]*pow(2,genomeLength-2-j-1)
      z1=0;
       for j in range(1,genomeLength):
           z1=z1+chromosome[i,j]*pow(2,genomeLength-j-1)
      z2=0
      for j in range(1, genomeLength-1):
        z2=z2+chromosome[i,j+genomeLength]*pow(2,genomeLength-1-j-1)
      z3=0
      for j in range(1, genomeLength-2):
        z3=z3+chromosome[i,j+genomeLength]*pow(2,genomeLength-2-j-1)
      z1=0;
       for j in range(1,genomeLength):
           z4=z4+chromosome[i,j]*pow(2,genomeLength-j-1)
      z2=0
      for j in range(1, genomeLength-1):
        z5=z5+chromosome[i,j+genomeLength]*pow(2,genomeLength-1-j-1)


      ctrl_pts=[x1,x2,x3,y1, y2, y3, z1, z2, z3, z4, z5] # ctrl_points we need
      trajectory=curve_3D_trajectory(flight_nr, ctrl_pts)
      fitness[i]= -compute_cost(trajectory) # - because we want t minimalize the cost

      
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
    print("fitness sum = ",fitness_total)

#########################################################
# QUANTUM ROTATION GATE                                 #
#########################################################
def rotation():
    rot=np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(1,popSize):
       for j in range(1,genomeLength):
           if fitness[i]<fitness[best_chrom[generation]]:
             # if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:
               if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==1:
                   # Define the rotation angle: delta_theta (e.g. 0.0785398163)
                   delta_theta=0.0785398163
                   rot[0,0]=math.cos(delta_theta); rot[0,1]=-math.sin(delta_theta);
                   rot[1,0]=math.sin(delta_theta); rot[1,1]=math.cos(delta_theta);
                   nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                   nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                   qpv[i,j,0]=round(nqpv[i,j,0],2)
                   qpv[i,j,1]=round(1-nqpv[i,j,0],2)
               if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==0:
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
    Fitness_evaluation(generation)
    while (generation<generation_max-1):
      rotation()
      mutation(0.01,0.001)
      generation=generation+1
      Measure(0.5)
      Fitness_evaluation(generation)


