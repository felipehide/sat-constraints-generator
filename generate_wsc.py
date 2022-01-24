#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weighted SAT generator 
"""

import random
import numpy as np
import sys
import os
import math
from scipy.stats import truncnorm

def write_formula2file(fname, formula):
    """
    Writes the generated formula to a .txt file
    Each clause is printed in an individual line
    
    Inputs:
        fname: string
            Name of the file to be saved
        formula: array
            The formula to be saved
    Outputs:
        None
    """
    with open(fname, 'w') as f:
        for c in formula:
            f.write("".join(str(c).strip('[').strip(']').split(",")))
            f.write("\n")
            
    print(formula)
    

def generate_clauses_set(seed,fname,n,m,k = 3, negation=True):
    """
    Generates the SAT formula
        The variables are represented by their numbers, from 1 to "n"
        if a variable is negated, then the index appears negative 
        for example, a clause (X1, ^X5, X3) will be [1, -5, 3]
        Use sets because it is much easier to check pertenence
    
    Inputs:
        seed: int
            The random seed, assigned for repeatability
        fname: string
            The filename to save the generated formula
        n: int
            Number of variables
        m: int
            Number of clauses
        k: int
            Number of literals per clause
        negation: bool
            Whether the literals can be negated or not
    Outputs:
        formula
    """
    
    random.seed(seed)
    
    formula_set = set([])
    i=0
    
    while i < m:
        c_set=set([])
        ja = 0
        while ja < k:
            r = random.randint(1,n)
            #If negation (parameter for the method) is on, 
            #    with probability 0.5 negate a variable
            #    negation is denoted by making r negative
            if negation == True:
                pr_n = random.random()
                if pr_n <=0.5:
                    r = -1*r
                    #print(r)
            if r not in c_set and -r not in c_set:
                c_set.add(r)
                ja+=1      
        
        if c_set not in formula_set:
            # to insert a set with in another set it has to be frozenset
            formula_set.add(frozenset(c_set))
            i = i+1
    # convert to list
    formula = []
    for x in list(formula_set):
        formula.append(list(x))
    # sort list using al x[i] as key (x[1], x[2], ...)
   # formula.sort(key=lambda x: tuple(( x[i] for i in range(k))))
    write_formula2file(fname, formula)
    return formula

def read_clauses(fname):
    """
    Reads a formula from a given file "fname"
    Written the same way write_formula2file() saves it, each clause at a line

    Args:
        fname: string 
            Name of the file to be read, on the same folder.

    Returns:
        formula: list
            Formula read from the file
    """
    
    formula = []
    with open(fname) as fp:
       for cnt, line in enumerate(fp):
           clause = []
           #print("Line {}: {}".format(cnt, line))
           #print("".join(line))
           str_list = line.strip('\n').split(" ")
           clause = [ int(x) for x in str_list]
           #print(clause)
           formula.append(clause)
    #print(formula)
    return(formula)

def write_sat_constraints2file(fname, Nv, Nc, formula, weights, eq_const, ineq_const, ineq_G  ):
    """
    Writes the file "sat_constraints", which contains the formula, weights, eq. and ineq.

    Args:
        fname: string
            Name of the file to be read
        Nv: int
            Number of variables
        Nc: int
            Number of clauses
        formula: array
            formula to be saved
        weights: arry
            weights of each clause
        eq_const: array
            clauses that are in the eq. constraints
        ineq_const: array
            clases that are in the ineq. constraints
        ineq_G: array
            Right hand side of the ineq. constraints

    Returns:
        None.

    """    
    fa = open("clauses_"+fname, "w")
    
    with open(fname, 'w') as f :
        #N C  
        f.write( str(Nv) + " " + str(Nc) + "\n" )
        #P Q
        f.write( str(len(eq_const)) + " " + str(len(ineq_const)) + "\n" )

        #f.write("# Equality constraints \n")
        #f.write("# h_i n w la lb lc ... w la lb lc 0 \n")              
        i = 0
        for const_k in eq_const:
            # To verify what clauses are included in the constraint 
            fa.write("h"+str(i+1)+" ")
            fa.write("".join(str(const_k).strip('[').strip(']').split(",")) + "\n")
            # Real output
            f.write(str(len(const_k))+" ")
            for c in const_k:
                idx = c-1
                f.write(str(weights[idx]) + " ")
                f.write("".join(str(formula[idx]).strip('[').strip(']').split(","))+" ")
            f.write("0\n")
            i += 1

                
#        f.write("# Inequality constraints \n")
#        f.write("# h_i n w la lb lc ... w la lb lc G \n")              
        i = 0
        for const_k in ineq_const:
            # To verify what clauses are included in the constraint 
            fa.write("g"+str(i+1)+" ")
            fa.write("".join(str(const_k).strip('[').strip(']').split(",")) + "\n")
            # Real output
            f.write(str(len(const_k))+" ")
            for c in const_k:
                idx = c-1
                f.write(str(weights[idx]) + " ")
                f.write("".join(str(formula[idx]).strip('[').strip(']').split(","))+" ")
            f.write(str(ineq_G[i])+"\n")
            i += 1
        f.write("\n")
        
        f.write("\n")
        f.write("File format \n")
        f.write("N C\n")
        f.write("P Q\n")
        f.write("n w a b c ... w a b c G=0 \n")        
        f.write("...\n") 
        f.write("n w a b c ... w a b c G>0 \n")        
        f.write("...\n") 
        f.write("\n") 
        f.write("Symbols \n")
        f.write("N: number of variables \n")
        f.write("C: number clauses \n")
        f.write("P: number of equality constraints \n")
        f.write("Q: number of inequality constraints \n")
        f.write("n: number of clauses in the constraint \n")        
        f.write("w: weight of a clause \n")  
        f.write("a, b, c: literals in a clause \n")        
        f.write("G: right side of the constraint, \n")
        f.write("   G = 0 in equality constraints  \n")
        f.write("   G > 0 in inequality constraints  \n")
        f.write("\n")
        f.write("weights: " + str(weights)+"\n")
        f.write("ineq G : " + str(ineq_G)+"\n")
              
    fa.close()
                    
# Generates rounding to round_perc (from 0 tp 100)
def sample_weights_normal_distribution_rounding(sample_size=30, seed=1, round_value = 0, round_perc=0):
    # Get 100000 random numbers normally distributed with mean 0 and 5Stdev N(0,5Stdev)
    # If you sum up all the probabilities in this interval you approach one 1
    # The numbers you get are in the range -10 to 10
    scale = 5.
    ranged = 10
    size = 100000
    amplify = 50

    X = truncnorm(a=-ranged/scale, b=+ranged/scale, scale=scale).rvs(size=size, random_state=seed)
    #Get the firts 10 numbers
    Xsample = (X*amplify).round().astype(int)[0:sample_size]
    

    if (round_value != 0):
        for i in range(0,int(len(Xsample)*(round_perc/100))):
            Xsample[i] = int(math.ceil(Xsample[i]/round_value))*round_value
    Xsample = [i+amplify*ranged for i in Xsample]
    Xsample.sort()  
    # shift the number to be in the range 0 1000 with mean 500
    
    return list(Xsample)

def assign_clauses_evenly(Nc, P, Q, Debug):
       
    constraints = []
    c_ids = [i for i in range(1,Nc+1)]
    c_ids = np.random.permutation(range(1,Nc+1))
    if Debug:        
        print("Shuffled clauses ids for assigments to constraints according to the cuts")
        print(c_ids)
    AveCC = int(Nc/(P+Q))
    if Debug:
        print("Base number of clauses per constraint", AveCC)
        print("No yet assigned clauses", Nc%(P+Q))
        print("Cuts for assigments")
    j = 0
    for i in range(0,P+Q):
        k = j + AveCC 
        if i >= P+Q - Nc%(P+Q) :
                k += 1
        if Debug:
            print(j,k)
        c_i =list(c_ids[j:k])
        constraints.append(c_i)
        j = k
    if Debug:
        print("Constraints before Overlap")
        print(constraints)
        print(type(constraints))
    # Verify that all clauses have been assigned to a constraint
    l = 0
    for c in constraints:
        l += len(c)
    if Debug:
        print("Number of clauses = ", Nc, " Assigned clauses to constraints = ", l)
    if l != Nc:
        print("Error, all clauses must appear in the constraints ")

    cons_ids = np.random.permutation(range(0,P+Q))

    if Debug:
        print("Shuffled constraints ids")
        print(cons_ids)
        
    # Make sure that inequality constraints are made up of 
    # at least 2 or more clauses
    n_ineq = 0
    eq_constraints = []
    ineq_constraints = [] 
    for i in range(0,P+Q):
        if n_ineq < Q:
            if len(constraints[cons_ids[i]]) > 1:
                ineq_constraints = ineq_constraints + [constraints[cons_ids[i]]]
                n_ineq += 1
            else:
                eq_constraints = eq_constraints + [constraints[cons_ids[i]]]
        else:
            eq_constraints = eq_constraints + [constraints[cons_ids[i]]]
    
    if len(eq_constraints) != P or len(ineq_constraints) != Q:
        print("Too many one-clause constrainst")
        print("We cannot create the requiered number of inequality constraints")
        print("Equality constraints:   requiered ", P, " created ", len(eq_constraints))
        print("Inequality constraints: requiered ", Q, " created ", len(ineq_constraints))
        sys.exit()
    if Debug:    
        print("Equality Constraints")
        for i in range(0,P):
            print(eq_constraints[i])
        print("Inequality Constrains")
        for i in range(0,Q):
            print(ineq_constraints[i])
            
    return eq_constraints,  ineq_constraints

def assign_clauses_specify(Nc, P, Q, CinP, CinQ, seed, Debug): 
    #Debug = True
    eq_constraints = []
    ineq_constraints = []
    c_ids = [i for i in range(1,Nc+1)]
    
    np.random.seed(seed)
    c_ids = np.random.permutation(range(1,Nc+1))
    
    if Debug:        
        print("Shuffled clauses ids for assigments to constraints ")
        print(c_ids)

    j = 0
    k = 0
    for i in range(0,P):
        k = j + CinP[i]
        if Debug:
            print(j,k)
        c_i =list(c_ids[j:k])
        eq_constraints.append(c_i)
        j = k
    if Debug:
        print("Equality constraints")
        print(eq_constraints)

    # k keeps the number of clauses assigned so far
    j = k
    
    for i in range(0,Q):
        k = j + CinQ[i]
        if Debug:
            print(j,k)
        c_i =list(c_ids[j:k])
        ineq_constraints.append(c_i)
        j = k
    if Debug:
        print("Inequality constraints")
        print(ineq_constraints)

    # Verify that all clauses have been assigned to a constraint
    l = 0
    for c in eq_constraints + ineq_constraints:
        l += len(c)
    if Debug:
        print("Number of clauses = ", Nc, " Assigned clauses to constraints = ", l)
    if l != Nc:
        print("Error, all clauses must appear in the constraints ")
    
    return eq_constraints, ineq_constraints    

# Add additional clauses to constraints to allow overlap: 
# i.e. the same clause is assidned to more than one constraint
# Select a constraint randomly
# Add a clause randomly making sure that 
# the same clause does not appear twice in the same constraint
def add_overlapping_clauses(eq_constraints, ineq_constraints, 
                            Nc, Nov, P, Q, Debug):
    for i in range(Nov):
        # id of the constraint selected randomly
        j = random.randint(0,P+Q-1) 
        # Generate all possible clause ids
        c_ids = [k for k in range(1,Nc+1)]
        # chose the constraint where to add an averlapping clause
        if j < P:
            ctype = "Equality"
            constraints = eq_constraints
        else:
            ctype = "Inequality"
            constraints = ineq_constraints
            j = j-P
        # Remove those clauses ids already present in the constraint
        for c in constraints[j] :
            c_ids.remove(c)
        # id of clause to be added to the cosntraint is selected randomly
        # get a random position in the list 
        k = random.randint(0,len(c_ids)-1)
        if Debug:            
            print(ctype +" Constraint id ", j, " add clause", c_ids[k])
        # add the c_id in the selected position
        constraints[j] = constraints[j] + [c_ids[k]]
        constraints[j].sort()
    eq_constraints.sort()
    ineq_constraints.sort()
    if Debug:
        print("Overlap", Nov)
        print("Constraints after Overlap")
        print(eq_constraints)
        print(ineq_constraints)
    return eq_constraints, ineq_constraints


def ineq_rigth_hand_side(Q, weights, ineq_constraints, w_point, round_value, Debug):
    """
    Generates the right hand side of the inequalities
    
    A non-satisfied constraint has its weight added up
    So a stronger weighting point (w_point) demands a smaller right hand side
    
    Inputs: 
        Q: int
            number of inequality constraints
        weights: list
            list of weights of each clause
        ineq_constraints: list
            list with which clauses are assigned to ineq. constranints
        w_point: int
            Value from 0 to 100 which assigns the strength of the ineq. const.
    Outputs: 
        ineq_G: list
            Right hand side values
    """
    
    Debug = True    
    if Debug:
        print("Rigth hand side of inequalities")
        
    # contains the right hand side of the ineq. const. clauses
    ineq_G = []
    
    # Loops through all ineq. const. clauses
    for i in range(0,Q):
        cons = ineq_constraints[i] # Current clause being worked on
        if Debug:            
            print("---")
            print(cons)
            
        const_W = [] # Contains the weights of the ineq. const. clauses
        const_G = 0 # The resulting right hand side value        
        weights_sum = 0 # The sum of the weight of the current clause
        
        for j in range(len(cons)):
            k = cons[j]  # clause number
            wid = k - 1 # Weight of the current clause k
            
            if Debug:
                print(k, weights[wid])
                
            const_W = const_W + [weights[wid]] # Appends the weight of the current loop
            
        weights_sum = sum(const_W) # Calculates the sum of the weights
            
        cutting_point = w_point/100
        const_G = int(cutting_point*weights_sum)
        if (round_value != 0):
            const_G = int(math.ceil(const_G/round_value))*round_value
                        
        if Debug:
            print("W ", const_W, " Tw ", sum(const_W), " G ", const_G)
            
        ineq_G = ineq_G + [const_G] 
        
    if Debug:
        print("Inequalities right hand side")
        print(ineq_G)
        
    return(ineq_G)

def ineq_rigth_hand_side_per_clause(Q, weights, ineq_constraints, w_point, Debug):
    """
    Generates the right hand side of the inequalities by using a rounded 
    percentage of number of clauses
    
    A non-satisfied constraint has its weight added up
    So a stronger weighting point (w_point) demands a smaller right hand side
    
    Inputs: 
        Q: int
            number of inequality constraints
        weights: list
            list of weights of each clause
        ineq_constraints: list
            list with which clauses are assigned to ineq. constranints
        w_point: int
            Value from 0 to 100 which assigns the strength of the ineq. const.
    Outputs: 
        ineq_G: list
            Right hand side values
    """
    
    Debug = True    
    if Debug:
        print("Rigth hand side of inequalities")
        
    # contains the right hand side of the ineq. const. clauses
    ineq_G = []
    
    # Loops through all ineq. const. clauses
    for i in range(0,Q):
        cons = ineq_constraints[i] # Current clause being worked on
        if Debug:            
            print("---")
            print(cons)
            
        const_W = [] # Contains the weights of the ineq. const. clauses
        const_G = 0 # The resulting right hand side value        
        weights_sum = 0 # The sum of the weight of the current clause
        
        for j in range(len(cons)):
            k = cons[j]  # clause number
            wid = k - 1 # Weight of the current clause k
            
            if Debug:
                print(k, weights[wid])
                
            const_W = const_W + [weights[wid]] # Appends the weight of the current loop
        
        cutting_point = int(len(cons)*(w_point/100))
        const_G = sum(const_W[0:cutting_point])
        
        # weights_sum = sum(const_W) # Calculates the sum of the weights
            
        # const_G = int(cutting_point*weights_sum)
                        
        if Debug:
            print("W ", const_W, " Tw ", sum(const_W), " G ", const_G)
            
        ineq_G = ineq_G + [const_G] 
        
    if Debug:
        print("Inequalities right hand side")
        print(ineq_G)
        
    return(ineq_G)

# OLD GENERATOR THAT TOOK A PERCENTAGE OF THE NUMBER OF WEIGHTS
# Generate rigth hand side of inequalities
# Weighting point w_point
#   U Uniform  rand(1,|W|-1)
#   S Strong (lo the left side rand(1,int(|W|/2)))
#   W Weak rand(int(|W|/2), W-1)
def OLD_ineq_rigth_hand_side(Q, weights, ineq_constraints, w_point, Debug):
    Debug = True    
    if Debug:
        print("Rigth hand side of inequalities")
    ineq_G = []
    for i in range(0,Q):
        cons = ineq_constraints[i]
        if Debug:            
            print("---")
            print(cons)
        const_W = []
        const_G = 0
        for j in range(len(cons)):
            k = cons[j]  #clause number
            wid = k - 1
            if Debug:
                print(k, weights[wid])
            const_W = const_W + [weights[wid]]
        # if there are two clauses, take the min weigth
        if len(const_W) == 2:
            const_G = min(const_W)
            nw_G = -1
        else:
            # randomly get the weighting point in the array of weigths
            #   Uniform  rand(1,|W|-1)
            if w_point == "Uniform":
                nw_G = random.randint(1,len(const_W)-1)
            #   Strong (lo the left side rand(1,int(|W|/2)))
            if w_point == "Strong":
                print(const_W,len(const_W)/2,int(len(const_W)/2))
                nw_G = random.randint(1,int(len(const_W)/2))
            #   Very Strong (lo the left side rand(1,int(|W|/4)))
            if w_point == "VeryStrong":
                print(const_W,len(const_W)/4,int(len(const_W)/4))
                nw_G = random.randint(1,int(len(const_W)/4))
            if w_point == "80Strong":
                print(const_W,len(const_W)*2/10,int(len(const_W)*2/10))
                nw_G = random.randint(1,int(len(const_W)*2/10))
            if w_point == "VeryVeryStrong":
                print(const_W,len(const_W)/10,int(len(const_W)/10))
                nw_G = random.randint(1,int(len(const_W)/10))
            #   Weak rand(int(|W|/2), W-1)
            if w_point == "Weak":
                nw_G = random.randint(int(len(const_W)/2),len(const_W)-1) 
#            nw_G = random.randint(1,len(const_W)-1)
            const_G = sum(const_W[0:nw_G]) 
        if Debug:
            print("W ", const_W, " Tw ", sum(const_W), " i_G ", nw_G, " G ", const_G)
        ineq_G = ineq_G + [const_G] 
    if Debug:
        print("Inequalities right hand side")
        print(ineq_G)
    return(ineq_G) 

# Manipulate weights so we can have active constraints
def manipulate_weights(Q, weights, ineq_G, ineq_const):
    
    new_unsorted_weights = []
    
    # Sorts the weights in order of the constraint
    const_W = [] # Contains the weights of the ineq. const. clauses
    
    for j in range(len(weights)):
        k = ineq_const[0][j]  # clause number
        wid = k - 1 # Weight of the current clause k
        
        const_W = const_W + [weights[wid]] # Appends the weight of the current loop
    
    clause_group = int(len(weights)*0.1) # Groups clauses in 10% of the Nc to try to force active constraints
    for i in range(0, int(len(weights)/clause_group)):
        weight_group = const_W[i*clause_group:i*clause_group+clause_group]
        print("weight group: " + str(weight_group)+"; sum: " + str(sum(weight_group)))
        weight_sum = sum(weight_group)
        
        if (weight_sum!=ineq_G[0]):
            weight_diff = ineq_G[0] - weight_sum
            # print("weight diff: " + str(weight_diff))
            dist_weight_diff = int(weight_diff/len(weight_group))
            # print("dist diff: " + str(dist_weight_diff))
            weight_group = [i+dist_weight_diff for i in weight_group]
            weight_group = [i if i>1 else 1 for i in weight_group] # Changes negative weights to 1
            new_diff = ineq_G[0] - sum(weight_group)
            if (new_diff>0):
                weight_group[weight_group.index(min(weight_group))] += new_diff 
            if (new_diff<0):
                weight_group[weight_group.index(max(weight_group))] += new_diff 
            
            print("new new weight group: " + str(weight_group)+"; sum: " + str(sum(weight_group)))
    
        new_unsorted_weights = new_unsorted_weights + weight_group
        new_unsorted_weights.sort()
    
    # Resorts the weights
    return new_unsorted_weights
    # print(new_weights)
            
    
# def generator_wsc_instances
def generate_wsc(Ninstances, Nv, Nc, P, Q, Wpoint, Assignment, 
                                 CinP=[], CinQ=[], Nov=0, method='none', round_value=0, round_perc=0):
    """
    Ninstances int: number of instannces
    Nv int : number of variables
    Nc int : number of clauses
    P  int : number of equality constraints
    Q  int : number of inequality constraints
    Wpoint int : weighting point: from 0 to 100
    Assigment str: assigment of clauses : 'Evenly', 'Specify' 
    CinP list : number of clauses in equality constraints if A = Specify
    CinQ list : number of clauses in inequality constraints if A = Specify
    Nov int : number of overlapping clauses 
    """
    
    CinP_str = ""
    for i in CinP:
        CinP_str += str(i) + "_"
    CinQ_str = ""
    for i in CinQ:
        CinQ_str += str(i) + "_"

    folder_name = 'WSC_N' + str(Nv) + '_C' + str(Nc) + '_P' + str(P) + \
            '_Q' + str(Q) + '_W' + str(Wpoint) + '_A' + Assignment[0] + \
            '_O' + str(Nov) 
            
    if method=='rounding':
        folder_name += "_r" + str(round_value) + "_r" + str(round_perc)
    os.mkdir('../problems/WSC/' + folder_name)
    os.chdir('../problems/WSC/' + folder_name)
    Debug = False
    for seed in range(1,Ninstances+1):
        # generate_sat_constraints(fname, Nv, Nc, P, Q, Wpoint, Assignment, 
        #                          CinP, CinQ, Nov, seed, False)
        
        # Step 1 Generate Nc cluases randomly
        fname = "sat.txt"
        formula = generate_clauses_set(seed, fname, Nv, Nc)
        if Debug :
            print("--- Generate sat constraints ---")
            print("Sat formula ")
            print(formula)
            
        # Step 2: Generate weights based on normal distribution
        weights = sample_weights_normal_distribution_rounding(Nc,seed,round_value,round_perc)
                    
        if Debug:
            print("Weights")
            print(weights)
            
        if Assignment == "Evenly":
            eq_const, ineq_const = assign_clauses_evenly(Nc, P, Q, Debug)
        elif Assignment == "Specify":
            eq_const, ineq_const = assign_clauses_specify(Nc, P, Q, CinP, CinQ, seed, Debug)    
            
        eq_const, ineq_const = add_overlapping_clauses(
                eq_const, ineq_const, Nc, Nov, P, Q, Debug)
    
        print(Q)
        ineq_G=[]
        if Q > 0:
            ineq_G = ineq_rigth_hand_side(Q, weights, ineq_const, Wpoint, round_value, Debug)    
        
        # Manipulates the weights
        # weights = manipulate_weights(Q, weights, ineq_G, ineq_const)
        
        fname_const_sat = folder_name + '_I' + str(seed) + '.txt'
        
        write_sat_constraints2file(fname_const_sat, Nv, Nc, formula, weights, 
                                    eq_const, ineq_const, ineq_G)
        if Debug:
            print("---End sat constraints generation ---")
        # write constraints to a file

# Example call to the generator:
# Ni = number of instances
# Nv = Number of variables
# Nc = Number of clauses (sum(CinP) + sum(CinQ))
# P  = Number of equality constraints
# Q  = Number of inequality constraints
# Wp = Ratio to canculate the right hand side of inequality constraints 
# Assignment = "Evenly" to split clauses evenly, "Specify" to specify the number of clauses of each constraint
# CinP = Array with number of clauses in each equality constraint
# CinQ = Array with number of clauses in each inequality constraint
# Ov = To add overlap between clauses
# gwsc.generate_wsc(Ni, Nv, Nc, P, Q, Wp, Assignment, CinP, CinQ, Ov)
