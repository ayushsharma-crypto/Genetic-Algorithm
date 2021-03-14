#!/usr/bin/env python
# coding: utf-8

# In[1]:


import client as server
import numpy as np
import os
import json
import random


# In[2]:


TEAM_ID = 'colthAUIKUTfdh4qWrnHBhJzkyEm8kt4qIue1BKtyvLItfp8Po'
DEFAULT_INITIAL_OVERFIT_VECTOR = [
    0.0, 
    -1.45799022e-12, 
    -2.28980078e-13,  
    4.62010753e-11, 
    -1.75214813e-10, 
    -1.83669770e-15,  
    8.52944060e-16,  
    2.29423303e-05, 
    -2.04721003e-06, 
    -1.59792834e-08,  
    9.98214034e-10
]
MAX_GENE_VAL = 10
MIN_GENE_VAL = -10
POPULATION_SIZE = 10
GENERATION_LOOP = 10
TRACE = "trace.json"
MUTATION_PROBABILITY = 0.6


# In[3]:


def is_existing_file(fileName):
    '''
    This funtion returns True if 
    filename exist  otherwise False
    '''
    return os.path.exists(fileName)


# In[4]:


def is_valid_file(filename):
    '''
    This function checks existence of file
    filename and if exist then checks if it
    empty or not.
    '''
    if is_existing_file(filename):
        return (os.stat(filename).st_size != 0)
    else:
        return False


# In[5]:


def read_file(filename):
    '''
    This function will read the filename
    and return it's content.
    '''
    with open(filename,'r') as read_file:
        data=json.load(read_file)
    return data


# In[6]:


def write_file(filename,data):
    '''
    This function will write data in the filename
    and return final content of it.
    '''
    with open(filename,'w') as write_file:
        json.dump(data, write_file, indent = 4)
    return read_file(filename)


# In[7]:


def get_both_err(population):
    '''
    This function utilises the API 
    call provided to us for getting the
    errors on the vectors within the population.
    
    Parameter
    ---------
    population: list of vector of 11-D
    
    Return
    ------
    It returns two list train_err & validation_err
    which are errors for the given poplation's vectors.
    '''
#     train_err = [ random.randint(1,300) for i in range(len(population))]
#     validation_err = [ random.randint(1,300) for i in range(len(population))]
    
    train_err = []
    validation_err = []
    for individual in population:
        [te, ve]= server.get_errors(TEAM_ID,individual)
        train_err.append(te)
        validation_err.append(ve)
    return train_err, validation_err


# In[8]:


def get_fitness(te, ve):
    '''
    This function calculates the fitness
    for given list of errors. Higher the
    fitness more fit/perfect the vector.
    Returns the list of fitness for 
    corresponding errors.
    '''
    fitness = []
    for i in range(len(te)):
        sum_err = te[i] + ve[i]
        abs_diff_err = abs(te[i] - ve[i])
        fit = ( sum_err) + ( abs_diff_err)
        fit = 1/fit
        fitness.append(fit)
    return fitness


# In[9]:


def selection_percentage(fitness):
    '''
    This function will return the list for percentage
    of chance of being selected for cross-over.
    '''
    fit_ness = np.array(fitness)
    total = sum(fit_ness)
    perc = list(map(lambda fit_val: ((100*fit_val)/total), fitness))
    return perc


# In[10]:


def create_fitness(pop):
    '''
    It takes population which is an array
    of POPULATION_SIZE 11-D vectors. And
    return POPULATION_SIZE 15-D vectors.
    In which last 5 columns will be te, ve,
    fitness and selection percentage.
    '''
    te, ve = get_both_err(pop)
    fitness = get_fitness(te, ve)
    percentage = selection_percentage(fitness)
    pop_fitness = np.column_stack((pop,te, ve, fitness, percentage))
    return pop_fitness


# In[11]:


def select_population(pop_fit):
    '''
    This function will select the vectors
    based on their selection percentage and
    return it's np list.
    '''
    fitness_perc = np.copy(pop_fit[:,-1:])
    fitness_perc_list = []
    for item in fitness_perc:
        fitness_perc_list.append(item[0]/100)
    selected_index = np.random.choice(POPULATION_SIZE, POPULATION_SIZE, fitness_perc_list)
    select_pop = []
    for index in selected_index:
        select_pop.append(pop_fit[index].copy())
    return np.array(select_pop)


# In[12]:


def cross_over(p1,p2):
    '''
    This function simply does the cross-over
    on two individual p1,p2 and returns c1,c2
    i.e. crossed-child.
    '''
    crossover_point = random.randint(1, 10)
    c1 = list(p1[:crossover_point]) + list(p2[crossover_point:])
    c2 = list(p2[:crossover_point]) + list(p1[crossover_point:])
    return c1, c2

def simulate_cross_over(selected_pop):
    '''
    This function will perform the cross-over
    on selected_population and generate POPULATION_SIZE
    total childs/individual.
    '''
    selected_vector = np.copy(selected_pop[:,:-4])
    cross_detail = []
    crossed_pop = []
    for i in range(POPULATION_SIZE//2):
        r1 = random.randint(0,POPULATION_SIZE-1)
        r2 = random.randint(0,POPULATION_SIZE-1)
        p1 = selected_vector[r1]
        p2 = selected_vector[r2]
        c1, c2 = cross_over(p1,p2)
        crossed_pop.append(c1)
        crossed_pop.append(c2)
        cross_detail.append(np.array([c1, c2, p1, p2]))
    return np.array(crossed_pop), np.array(cross_detail)


# In[13]:


def mutation(crossed_popuplation):
    '''
    This function will perform gene mutation
    on the population resulted from cross-over.
    '''
    mutated_population = []
    crossed_pop = np.copy(crossed_popuplation)
    for i in range(len(crossed_pop)):
        curr_vec = crossed_pop[i]
        for j in range(len(curr_vec)):
            if random.uniform(1,10)<=(10*(MUTATION_PROBABILITY)/2):
                if curr_vec[j]==0:
                    curr_vec[j] = random.uniform(-0.01,0.01)
                else:
                    if j > 4:
                        fac = random.uniform(0, 1)
                    else:
                        fac = 1 + random.uniform(-0.03, 0.03)                        
                    new_gene = fac*curr_vec[j]
                    if abs(new_gene)<10:
                        curr_vec[j]=new_gene
        mutated_population.append(curr_vec)
    return np.array(mutated_population)


# In[14]:


def create_start_population():
    '''
    This function will create a default
    start file for dumping our 10 best vector
    before any iteration of algorithm when 
    called if START doesnot exist. Returns
    the content in same file after completion
    '''
    start = []
    for i in range(POPULATION_SIZE):
        curr_vec = DEFAULT_INITIAL_OVERFIT_VECTOR.copy()
        for j in range(len(curr_vec)):
            if random.uniform(1,10)<=(10*MUTATION_PROBABILITY):
                if curr_vec[j]==0:
                    curr_vec[j] = random.uniform(-0.05,0.05)
                else:
                    if j <= 4:
                        fac = random.uniform(0, 1)
                    else:
                        fac = 1 + random.uniform(-0.05, 0.05)
                    new_gene = fac*curr_vec[j]
                    if abs(new_gene)<10:
                        curr_vec[j]=new_gene
        start.append(curr_vec)
    return start


# In[15]:


def data_breach():
    '''
    This function access the stored vectors in
    file TRACE. Returns last 10 vectors with their
    errors and fitness and selection percentage.
    '''
    trace_data = read_file(TRACE)
    reproduction = trace_data['Trace'][-1]['reproduction']
    pop = reproduction['child']
    pop_fit = reproduction['child_fitness']
    return pop, np.array(pop_fit)


# In[16]:


def save_stuff(pop_fit,selected,cross,cross_detail,mutated,new_pop_fit):
    '''
    This function will append all the data corressponding
    to one reproduction in the TRACE file.
    '''
    trace_data = read_file(TRACE)
    trace_data1 = trace_data['Trace']
    reproduction = {
        'parent': pop_fit[:,:-4].tolist(),
        'parent_fitness': pop_fit.tolist(),
        'selected': selected.tolist(),
        'cross': cross.tolist(),
        'cross_detail':cross_detail.tolist(),
        'mutated': mutated.tolist(),
        'child': new_pop_fit[:,:-4].tolist(),
        'child_fitness': new_pop_fit.tolist()        
    }
    appendable = {
        'reproduction' : reproduction
    }
    trace_data1.append(appendable)
    write_file(TRACE, trace_data)


# In[17]:


def GA():
    '''
    Main function to be called to run
    the implemented genetic algorithm.
    '''
    generation_loop = GENERATION_LOOP
    pop = []
    pop_fit = []
    
    if not is_valid_file(TRACE):
        pop = create_start_population()
        pop_fit = create_fitness(pop)
        data = {"Trace": []}
        write_file(TRACE,data)
    else:
        pop, pop_fit = data_breach()
    
    for generation in range(generation_loop):
        
        selected_pop = select_population(pop_fit)
        crossed_pop, cross_detail = simulate_cross_over(selected_pop)
        mutated_pop = mutation(crossed_pop)
        new_pop = mutated_pop.tolist()
        new_pop_fit = create_fitness(new_pop)
        
        save_stuff(pop_fit,selected_pop,crossed_pop,cross_detail,mutated_pop,new_pop_fit)
        
        pop = new_pop
        pop_fit = new_pop_fit


# In[488]:


if __name__ == '__main__':
    GA()


# In[490]:


tempv = [
    0.017327441699636698,
    -1.45799022e-12,
    -2.287648457407279e-13,
    4.62010753e-11,
    -1.75214813e-10,
    -1.3514608366044225e-15,
    8.5294406e-16,
    5.800836067949083e-06,
    -1.5955733902066742e-06,
    -2.394848192566377e-09,
    6.387409623174343e-10
]
# tempv = [
#     -0.010745550130705474,
#     -1.4263864333069658e-12,
#     -2.195002139233098e-13,
#     4.9639066198589243e-11,
#     -1.7647527535706964e-10,
#     -1.6612084167350688e-16,
#     1.969890609438984e-18,
#     4.111594565099592e-09,
#     -7.335949881251237e-08,
#     -3.217999641758488e-11,
#     9.699424244065757e-13
# ]
# tempv = [
#     -0.010352359899980625,
#     -1.4938708798725434e-12,
#     -2.1344537013801585e-13,
#     4.754606122495976e-11,
#     -1.7513765845712067e-10,
#     -2.9750369207778414e-20,
#     2.216842207753494e-20,
#     8.683381211976487e-15,
#     -6.662021179304016e-09,
#     -2.6599123253785434e-14,
#     3.026701227755296e-17
# ]
# tempv = [
#     -0.00924596659365373,
#     -1.494818150350766e-12,
#     -2.211532640173319e-13,
#     5.039657579364812e-11,
#     -1.6753978508012588e-10,
#     -9.080082588227132e-25,
#     7.257195414468184e-26,
#     4.605087763217074e-20,
#     -1.3071635934940838e-11,
#     -1.4401903358063287e-17,
#     2.7501251088023155e-21
# ]
server.submit(TEAM_ID,tempv)


# In[ ]:




