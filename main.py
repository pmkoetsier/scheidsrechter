import time
start_time = time.time()

import pandas as pd
import datetime as dt
import numpy as np
import pygad
import statistics
from multiprocessing import Pool

duur_wedstrijd = 1.5
niet_fluiten_voor_wedstrijd = 1
reistijd = 1
maxtefluitenperdag = 3
thuisclub = 'ODIK'
te_fluiten_teams = ['ODIK 5', 'ODIK 6', 'ODIK 7', 'ODIK A2', 'ODIK A3', 'ODIK B2', 'ODIK C2', 'ODIK C3', 'ODIK D1', 'ODIK D2', 'ODIK D3', 'ODIK D4', 'ODIK E1', 'ODIK E2', 'ODIK E3', 'ODIK E4', 'ODIK F1', 'ODIK F2', 'ODIK MW1' ]
te_fluiten_teams_senior = ['ODIK 5', 'ODIK 6', 'ODIK 7', 'ODIK MW1']
te_fluiten_teams_voor_junior = ['ODIK E1', 'ODIK E2', 'ODIK E3', 'ODIK E4', 'ODIK F1', 'ODIK F2']
teams_beschikbaar = ['ODIK 3', 'ODIK 4','ODIK 5', 'ODIK 6', 'ODIK 7',  'ODIK A1', 'ODIK A2', 'ODIK A3', 'ODIK MW1']
teams_beschikbaar_junior = ['ODIK A1', 'ODIK A2', 'ODIK A3']


#import data
wedstrijden = pd.read_excel(r'2020-2021 VELD NAJAAR - min.xlsx', sheet_name='Blad1', parse_dates=[['Wedstrijddatum','Aanvangstijd']])

#Thuis of uit, als thuis dan true
wedstrijden['Thuiswedstrijd'] = wedstrijden['Thuis team'].str.contains(thuisclub, na=True)

#toevoegen grenzen aan data
wedstrijden['fluiten_tot'] = np.where(wedstrijden['Thuiswedstrijd'] == False, wedstrijden['Wedstrijddatum_Aanvangstijd'] - dt.timedelta(hours=niet_fluiten_voor_wedstrijd) - dt.timedelta(hours=reistijd) - dt.timedelta(hours=duur_wedstrijd), wedstrijden['Wedstrijddatum_Aanvangstijd'] - dt.timedelta(hours=niet_fluiten_voor_wedstrijd) - dt.timedelta(hours=duur_wedstrijd))
wedstrijden['fluiten_vanaf'] = np.where(wedstrijden['Thuiswedstrijd'] == False, wedstrijden['Wedstrijddatum_Aanvangstijd'] + dt.timedelta(hours=reistijd) + dt.timedelta(hours=duur_wedstrijd), wedstrijden['Wedstrijddatum_Aanvangstijd'] + dt.timedelta(hours=duur_wedstrijd))

#Moet deze wedstrijd worden gefloten?
wedstrijden['fluiten'] = wedstrijden['Thuis team'].isin(te_fluiten_teams)

wedstrijden['tefluitensenior'] = wedstrijden['Thuis team'].isin(te_fluiten_teams_senior)
# selector voor wedstrijden
wedstrijdenberekenen = wedstrijden.loc[(wedstrijden['fluiten'] == True)].copy()



def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    x = 0
    fitness = 0
    start_time = time.time()
    #TODO: parallel processing implementeren
    for i in range(len(solution)):
        #Checken of het toegewezen team kan fluiten op dit moment
        wedstrijddag = wedstrijdenberekenen.iloc[i].Wedstrijddatum_Aanvangstijd
        scheidsrechtervol = teams_beschikbaar[int(solution[i])]

        scheidsrechterzelfspelen = wedstrijden.loc[(wedstrijden['Wedstrijddatum_Aanvangstijd'].dt.strftime('%Y-%m-%d') == wedstrijddag.strftime('%Y-%m-%d')) & (wedstrijden['Team'] == scheidsrechtervol)]

        #checken of jeugdteam ook jeugs fluit
        if scheidsrechtervol in teams_beschikbaar_junior and wedstrijdenberekenen.iloc[i].Team not in te_fluiten_teams_voor_junior:
            fitness += -10000

        # Als ze helemaal niet hoeven te spelen die dag
        if len(scheidsrechterzelfspelen.index) == 0:
            fitness += 0
        elif scheidsrechterzelfspelen.iloc[0].fluiten_tot >= wedstrijdenberekenen.iloc[i].Wedstrijddatum_Aanvangstijd or wedstrijdenberekenen.iloc[i].Wedstrijddatum_Aanvangstijd >= scheidsrechterzelfspelen.iloc[0].fluiten_vanaf:
            fitness += 0
        else:
            fitness += -10000

    wedstrijdenberekenen.loc[:, 'Scheidsrechter'] = solution

    scheidsrechterdatummatrix = pd.crosstab(wedstrijdenberekenen.Wedstrijddatum_Aanvangstijd.apply(lambda dt: dt.strftime('%Y-%m-%d')),wedstrijdenberekenen.Scheidsrechter)

    #Checken dat er niet te vaak wordt gefloten
    if scheidsrechterdatummatrix.values.max() > maxtefluitenperdag:
            fitness += -100000

    #Checken wat de variatie is tussen de teams in aantal te fluiten wedstrijden
    fitness += 1 / statistics.variance(scheidsrechterdatummatrix.sum())

    #variatie tussen senioren
    wedstrijdenseniorberekenen = wedstrijdenberekenen.loc[(wedstrijdenberekenen['tefluitensenior'] == True)]
    scheidsrechterseniorendatummatrix = pd.crosstab(
        wedstrijdenseniorberekenen.Wedstrijddatum_Aanvangstijd.apply(lambda dt: dt.strftime('%Y-%m-%d')),
        wedstrijdenseniorberekenen.Scheidsrechter)
    fitness += 1 / statistics.variance(scheidsrechterseniorendatummatrix.sum())

    return fitness

def fitness_wrapper(solution):
    return fitness_func(solution, 0)


class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool

        pop_fitness = pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

num_generations = 1600 # Number of generations.
num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 25 # Number of solutions in the population.
num_genes = int(wedstrijdenberekenen['fluiten'].count())

gene_space = list(range(len(teams_beschikbaar)))

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    last_fitness = ga_instance.best_solution()[1]

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = PooledGA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       gene_space=gene_space,
                       parent_selection_type="rank",
                       crossover_type="scattered"
                       )

# Running the GA to optimize the parameters of the function.

if __name__ == '__main__':
    with Pool(processes=16) as pool:
        ga_instance.run()


        # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
        #ga_instance.plot_fitness()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()



        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("--- %s seconds ---" % (time.time() - start_time))
    wedstrijdenberekenen['Scheidsrechter'] = solution
    wedstrijdenberekenen.to_excel('data.xlsx', sheet_name='scheidsrechters')