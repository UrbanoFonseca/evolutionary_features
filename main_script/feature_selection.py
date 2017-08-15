# Define the dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class evolutionary_features:
    # Applies evolutionary algorithm to feature selction by multiplying weights for each feature.
    
    def train(self, model, X, Y, method='predict', population=100, 
              hall_of_fame= 20, crossovers=40,  mutations=30, 
              generations=20):


        print('Coercing the data')
        # Make sure the data is coerce
        if population < hall_of_fame + crossovers + mutations:
            raise ValueError('The population must be greater than evolutions')

        if np.shape(X)[1] >= 10000:
            kf = StratifiedKFold(n_splits=3)
            train_index, test_index = kf.split(X, Y)
            x_train, x_test = X.iloc[train_index], Y.iloc[test_index]
            y_train, y_test = X.iloc[train_index], Y.iloc[test_index]
        else:
            x_train, x_test = X, X
            y_train, y_test = Y, Y

        # Convert percentage to integer
        for parameter in [hall_of_fame, crossovers, mutations]:
            if parameter < 1:
                parameter = int(parameter * population)


        print('Generating the first random sample')
        # This function creates a random set of weigths that will multiply by each feature of the data
        # Apply a genetic algorithm to the predict
        np.random.seed(1)
        w0 = np.random.rand(population, np.shape(X)[1]).tolist()

        # Create a record of the weights
        allGenes = pd.DataFrame(columns=['Genes','Result'])

        print('Testing the first sample')
        # individual is a vector of weights
        id = 0
        for individual in w0:
            new_x = individual * x_train
            # Cross validate the new generation
            allGenes.loc[id, 'Genes'] = individual
            allGenes.loc[id, 'Result'] = np.mean(cross_val_score(model, new_x, Y, cv=4))
            id += 1
        
        print('Starting point: ', max(allGenes['Result']))
        
        evolution = pd.DataFrame(columns=['Result'])
        evolution.loc['Start', 'Result'] = max(allGenes['Result'])
		
        for generation in range(1, generations + 1):
            # Generate a new generation
            allGenes = evolutionary_features.new_generation(allGenes,
                                                            population=population,
                                                            selection=hall_of_fame,
                                                            max_crossover=crossovers,
                                                            mutation=mutations)

            for individual in allGenes.index:
                weights = allGenes.loc[individual, 'Genes']
                new_x = weights * x_train
                # Cross validate the new generation
                allGenes.loc[individual, 'Result'] = np.mean(cross_val_score(model, new_x, Y, cv=4))
            print(generation, ' generation: ', max(allGenes['Result']))
            evolution.loc[generation, 'Result'] = max(allGenes['Result'])
        
        # Set the best estimator parameter as the best 
        self.best_estimator_ = allGenes.loc[allGenes['Result'].idxmax(), 'Genes']
        
        # Save the evolution
        self.evolution_ = evolution

    
    def crossover(mother, father, split=0.5):
        
        if np.shape(father) != np.shape(mother):
            print('The parents of the crossover function must have the same lenght')
        
        split = int(np.floor(split*len(father)))
        
        breed = mother[:split] + father[split:]
        
        return breed


    def new_generation(allGenes, selection=5, max_crossover = 10, mutation=10,
                       population=100):
        
        # Sort the data
        allGenes = allGenes.sort_values(by='Result', ascending=False)

        # Generates a new generation
        new_gen = pd.DataFrame(columns=['Genes', 'Result'])
        
        # SELECTION
        for hall_of_famer in range(0, selection):
            new_gen.loc[hall_of_famer, 'Genes'] = list(allGenes.iloc[hall_of_famer, 0])
        
        # CROSSOVER
        for breed in range(selection, max_crossover + selection):
            # Generate random indexes
            mum, pops = np.random.randint(0, selection, size=2)
            # Create the progenitors
            mother = allGenes.iloc[mum, 0]
            father = allGenes.iloc[pops, 0]
            child = pd.DataFrame(evolutionary_features.crossover(mother, father)).T
            # The child.values is a numpy ndarray
            # Convert to list (which returns a list of lists) and return fetch
            # its first list
            new_gen.loc[breed, 'Genes'] = child.values.tolist()[0]
        
        # MUTATION
        for mutant in range(max_crossover + selection,
                                   max_crossover + selection + mutation):
            id = mutant - (max_crossover + selection)
            new_gen.loc[mutant, 'Genes'] = (allGenes.iloc[id, 0] *  np.random.normal(loc=1, scale=0.05, size=(len(allGenes.iloc[id, 0])))).tolist()
        
        
        # NEW POPULATION
        for person in range(mutant + max_crossover + selection,
                            population):
            new_gen.loc[person, 'Genes'] = np.random.rand(1, np.shape(X)[1]).tolist()[0]
        
        
        allGenes = new_gen
        
        return allGenes
        
    
    
    def best_estimator_(self):
        self.best_estimator_ = []
        
    def fit(self, X):
        return self.best_estimator_ * X

    def evolution_(self):
        self.evolution = []
        
    def plot_evolution_(self):
        benchmark = self.evolution_.iloc[0,0]
        generations = len(self.evolution_) - 1
                         
        plt.plot((1, generations),(benchmark,benchmark), 'r--', label="benchmark")
        plt.plot(self.evolution_.iloc[1:, 0], 'g-', label="performance")     
        plt.legend(loc='best')                   
        plt.show()

    def output_best(self, X, output_folder, name='feature_selection_x'):
        best_x = self.best_estimator_ * X
        best_x.to_csv(output_folder + name + '.csv', sep=',', decimal='.')


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
