import datetime
import time
start_time = time.time()

import pandas as pd
import datetime as dt
import numpy as np
import pygad

duur_wedstrijd = 1.5
niet_fluiten_voor_wedstrijd = 1
reistijd = 1
thuisclub = 'ODIK'
te_fluiten_teams = ['ODIK 5', 'ODIK 6', 'ODIK 7', 'ODIK A2', 'ODIK A3', 'ODIK B2', 'ODIK C2', 'ODIK C3', 'ODIK D1', 'ODIK D2', 'ODIK D3', 'ODIK D4', 'ODIK E1', 'ODIK E2', 'ODIK E3', 'ODIK E4', 'ODIK F1', 'ODIK F2', 'ODIK MW1' ]
teams_beschikbaar = ['ODIK 3', 'ODIK 4','ODIK 5', 'ODIK 6', 'ODIK 7', 'ODIK A2', 'ODIK A3']

#import data
wedstrijden = pd.read_excel(r'2020-2021 VELD NAJAAR - min.xlsx', sheet_name='Blad1', parse_dates=[['Wedstrijddatum','Aanvangstijd']])

#Thuis of uit, als thuis dat True?
wedstrijden['Thuiswedstrijd'] = wedstrijden['Thuis team'].str.contains(thuisclub, na=True)

#toevoegen grenzen aan data
wedstrijden['fluiten_tot'] = np.where(wedstrijden['Thuiswedstrijd'] == False, wedstrijden['Wedstrijddatum_Aanvangstijd'] - dt.timedelta(hours=niet_fluiten_voor_wedstrijd) - dt.timedelta(hours=reistijd) - dt.timedelta(hours=duur_wedstrijd), wedstrijden['Wedstrijddatum_Aanvangstijd'] - dt.timedelta(hours=niet_fluiten_voor_wedstrijd) - dt.timedelta(hours=duur_wedstrijd))
wedstrijden['fluiten_vanaf'] = np.where(wedstrijden['Thuiswedstrijd'] == False, wedstrijden['Wedstrijddatum_Aanvangstijd'] + dt.timedelta(hours=reistijd) + dt.timedelta(hours=duur_wedstrijd), wedstrijden['Wedstrijddatum_Aanvangstijd'] + dt.timedelta(hours=duur_wedstrijd))

#Moet deze wedstrijd worden gefloten?
wedstrijden['fluiten'] = wedstrijden['Thuis team'].isin(te_fluiten_teams)

# selector voor wedstrijden
wedstrijdenberekenen = wedstrijden.loc[(wedstrijden['fluiten'] == True)].copy()

def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    x = 0
    fitness = 0
    for scheidsrechter in solution:

        wedstrijddag = wedstrijdenberekenen.iloc[x].Wedstrijddatum_Aanvangstijd
        scheidsrechtervol = teams_beschikbaar[int(solution[x])]
        scheidsrechterzelfspelen = wedstrijden.loc[(wedstrijden['Wedstrijddatum_Aanvangstijd'].dt.strftime('%Y-%m-%d') == wedstrijddag.strftime('%Y-%m-%d')) & (wedstrijden['Team'] == scheidsrechtervol)]

        if len(scheidsrechterzelfspelen.index) == 0:
            fitness += 1
        elif scheidsrechterzelfspelen.iloc[0].fluiten_tot >= wedstrijdenberekenen.iloc[x].Wedstrijddatum_Aanvangstijd or wedstrijdenberekenen.iloc[x].Wedstrijddatum_Aanvangstijd >= scheidsrechterzelfspelen.iloc[0].fluiten_vanaf:
            fitness += 1
        else:
            fitness += -10000


        x += 1

    return fitness

fitness_function = fitness_func

num_generations = 25 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 50 # Number of solutions in the population.
num_genes = int(wedstrijdenberekenen['fluiten'].count())
gene_space = [0,1,2,3,4,5,6]

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    last_fitness = ga_instance.best_solution()[1]

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       gene_space=gene_space)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()


wedstrijdenberekenen.loc[:,'Scheidsrechter'] = solution


print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("--- %s seconds ---" % (time.time() - start_time))