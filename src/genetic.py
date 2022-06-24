from src.base import BaseSplitter
from pygad import GA


class GeneticSplitter(BaseSplitter):
    def __init__(self, **kwargs):
        super(GeneticSplitter, self).__init__(**kwargs)
        self.df = None
        self.prots = None
        self.drugs = None

    @property
    def author(self):
        return "Roman"

    @staticmethod
    def get_all():
        return [
            GeneticSplitter(**{"G": 100, "P": 3, "KP": 2, "MP": 0.1, "CP": 0.2, "D": 1, "B": 1})
        ]

    @property
    def name(self):
        return "Genetic"

    def split_two(self, df, train_frac):
        super(GeneticSplitter, self).split_two(df, train_frac)
        self.df = df
        self.prots = df["Target_ID"].unique()
        self.drugs = df["Drug_ID"].unique()

        """Initialize the genetic algorithm based on some hyperparameter"""
        ga = GA(
            num_generations=self.params["G"],
            num_parents_mating=self.params["P"],
            fitness_func=lambda x, y: self._fitness_function_two(x, y, train_frac),
            sol_per_pop=10,
            num_genes=len(self.prots) + len(self.drugs),
            gene_type=int,
            keep_parents=self.params["KP"],
            crossover_probability=self.params["CP"],
            mutation_probability=self.params["MP"],
            mutation_by_replacement=True,
            gene_space=[0, 1],
        )
        ga.run()

        split = ga.best_solution()[0]
        train_solution = [bool(x) for x in split]
        test_solution = [not bool(x) for x in split]

        train_prots = self.prots[train_solution[:len(self.prots)]]
        test_prots = self.prots[test_solution[:len(self.prots)]]
        train_drugs = self.drugs[train_solution[len(self.prots):]]
        test_drugs = self.drugs[test_solution[len(self.prots):]]

        def assign(x):
            if x["Drug_ID"] in train_drugs and x["Target_ID"] in train_prots:
                return 'train'
            elif x["Drug_ID"] in test_drugs and x["Target_ID"] in test_prots:
                return 'test'
            else:
                return None
        df['split'] = df.apply(lambda x: assign(x), axis=1)
        df = df[df['split'].notna()]
        return df

    def _fitness_function_two(self, solution, idx, train_frac):
        """Evaluate the intermediate solution"""
        # split the data as suggested by the solution array
        train_solution = [bool(x) for x in solution]
        test_solution = [not bool(x) for x in solution]
        train_prots = self.prots[train_solution[:len(self.prots)]]
        test_prots = self.prots[test_solution[:len(self.prots)]]
        train_drugs = self.drugs[train_solution[len(self.prots):]]
        test_drugs = self.drugs[test_solution[len(self.prots):]]

        # check if any group of [train|test] [proteins|drugs] is empty -> Return minus infinity
        if any([len(x) == 0 for x in [train_prots, test_prots, train_drugs, test_drugs]]):
            return float("-inf")

        # extract the train, test, and dropped interactions
        drop_data = self.df[
            (self.df["Target_ID"].isin(train_prots) & self.df["Drug_ID"].isin(test_drugs)) |
            (self.df["Target_ID"].isin(test_prots) & self.df["Drug_ID"].isin(train_drugs))
        ]
        train_data = self.df[self.df["Target_ID"].isin(train_prots) & self.df["Drug_ID"].isin(train_drugs)]

        """
        actually compute the score to minimize the number of dropped interactions as well as the differences between 
        the rations of drugs, targets, and interactions between train set and test set.
        As the genetic algorithm is a maximization algorithm, we have to negate the minimization score
        """
        return - (
                self.params["D"] * (len(drop_data) / len(self.df)) +
                self.params["B"] * (
                        abs(len(train_data) / (len(self.df) - len(drop_data)) - train_frac) +
                        abs(len(train_drugs) / len(self.drugs) - train_frac) +
                        abs(len(train_prots) / len(self.prots) - train_frac)
                )
        )

    def split_three(self, df, train_frac, val_frac):
        super(GeneticSplitter, self).split_three(df, train_frac, val_frac)
        self.df = df
        self.prots = df["Target_ID"].unique()
        self.drugs = df["Drug_ID"].unique()

        """Initialize the genetic algorithm based on some hyperparameter"""
        ga = GA(
            num_generations=self.params["G"],
            num_parents_mating=self.params["P"],
            fitness_func=lambda x, y: self._fitness_function_three(x, y, train_frac, val_frac),
            sol_per_pop=10,
            num_genes=len(self.prots) + len(self.drugs),
            gene_type=int,
            keep_parents=self.params["KP"],
            crossover_probability=self.params["CP"],
            mutation_probability=self.params["MP"],
            mutation_by_replacement=True,
            gene_space=[0, 1, 2],
        )
        ga.run()

        split = ga.best_solution()[0]
        train_solution = [x == 2 for x in split]
        val_solution = [x == 1 for x in split]
        test_solution = [x == 0 for x in split]

        train_prots = self.prots[train_solution[:len(self.prots)]]
        val_prots = self.prots[val_solution[:len(self.prots)]]
        test_prots = self.prots[test_solution[:len(self.prots)]]
        train_drugs = self.drugs[train_solution[len(self.prots):]]
        val_drugs = self.drugs[val_solution[len(self.prots):]]
        test_drugs = self.drugs[test_solution[len(self.prots):]]

        def assign(x):
            if x["Drug_ID"] in train_drugs and x["Target_ID"] in train_prots:
                return 'train'
            if x["Drug_ID"] in val_drugs and x["Target_ID"] in val_prots:
                return 'val'
            if x["Drug_ID"] in test_drugs and x["Target_ID"] in test_prots:
                return 'test'
            return None

        df['split'] = df.apply(lambda x: assign(x), axis=1)
        df = df[df['split'].notna()]
        return df

    def _fitness_function_three(self, solution, idx, train_frac, val_frac):
        """Evaluate the intermediate solution"""
        # split the data as suggested by the solution array
        train_solution = [x == 2 for x in solution]
        val_solution = [x == 1 for x in solution]
        test_solution = [x == 0 for x in solution]
        train_prots = self.prots[train_solution[:len(self.prots)]]
        val_prots = self.prots[val_solution[:len(self.prots)]]
        test_prots = self.prots[test_solution[:len(self.prots)]]
        train_drugs = self.drugs[train_solution[len(self.prots):]]
        val_drugs = self.drugs[val_solution[len(self.prots):]]
        test_drugs = self.drugs[test_solution[len(self.prots):]]

        # check if any group of [train|test] [proteins|drugs] is empty -> Return minus infinity
        if any([len(x) == 0 for x in [train_prots, val_prots, test_prots, train_drugs, val_drugs, test_drugs]]):
            return float("-inf")

        # extract the train, test, and dropped interactions
        train_data = self.df[self.df["Target_ID"].isin(train_prots) & self.df["Drug_ID"].isin(train_drugs)]
        val_data = self.df[self.df["Target_ID"].isin(val_prots) & self.df["Drug_ID"].isin(val_drugs)]
        test_data = self.df[self.df["Target_ID"].isin(test_prots) & self.df["Drug_ID"].isin(test_drugs)]
        num_keep = len(train_data) + len(val_data) + len(test_data)
        num_dropped = len(self.df) - num_keep
        test_frac = (1 - train_frac - val_frac)
        """
        actually compute the score to minimize the number of dropped interactions as well as the differences between 
        the rations of drugs, targets, and interactions between train set and test set.
        As the genetic algorithm is a maximization algorithm, we have to negate the minimization score
        """
        score = - (
                self.params["D"] * (num_dropped / len(self.df)) +
                self.params["B"] * (
                        abs(len(train_data) / num_keep - train_frac) +
                        abs(len(val_data) / num_keep - val_frac) +
                        abs(len(test_data) / num_keep - test_frac) +
                        abs(len(train_drugs) / len(self.drugs) - train_frac) +
                        abs(len(val_drugs) / len(self.drugs) - val_frac) +
                        abs(len(test_drugs) / len(self.drugs) - test_frac) +
                        abs(len(train_prots) / len(self.prots) - train_frac) +
                        abs(len(val_prots) / len(self.prots) - val_frac) +
                        abs(len(test_prots) / len(self.prots) - test_frac)
                )
        )
        print(score)
        return score
