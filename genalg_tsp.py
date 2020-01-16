import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POPULATION_SIZE = 150  # Parzyste.
MUTATION_PROBABILITY = 0.05
NUMBER_OF_CITIES = 15  # Na razie ze względów alfabetycznych ograniczone do 26.
fig = plt.figure(figsize=(10, 6))


class WorldMap:
    """Klasa reprezentująca mapę (układ współrzędnych 1000x1000) z losowo utworzonymi miastami A, B, C..."""
    def __init__(self, size):
        self.city_pool = ""
        for i in range(size):
            self.city_pool += chr(65 + i)
        self.map = pd.DataFrame(np.random.choice(1000, (size, 2), replace=False),
                                index=list(self.city_pool), columns=['x', 'y'])
        self.distances = (self.map.x.apply(lambda q: (q - self.map.x) ** 2)
                          + self.map.y.apply(lambda q: (q - self.map.y) ** 2)).apply(np.sqrt)


class GenAlgTSP:
    """Klasa reprezentująca pojedynczą instancję algorytmu genetycznego i jego możliwe działania."""
    def __init__(self, number_of_cities, population_size, mutation_probability):
        self.number_of_cities = number_of_cities
        self.world_map = WorldMap(number_of_cities)
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.current_population = self.create_population()
        self.current_fitness = np.zeros_like(self.current_population)
        self.current_best_member = None
        self.current_worst_index = None
        self.selection()

    def create_population(self):
        """Zwraca losową populację chromosomów (stringów z literami, gdzie litera odpowiada miastu).
        Rozpoczynamy w mieście A."""
        city_pool = ""
        for i in range(1, self.number_of_cities):
            city_pool += chr(65 + i)
        population = np.empty(self.population_size, dtype=("U" + str(self.number_of_cities)))
        for i in range(self.population_size):
            population[i] = "A" + "".join(np.random.choice(list(city_pool), self.number_of_cities - 1, replace=False))
        return np.array(population)

    def route_distance(self, route):
        """Zwraca długość drogi dla pojedynczej trasy."""
        cities_list = list(route)
        distance = 0
        for i in range(self.number_of_cities - 1):
            distance += self.world_map.distances[cities_list[i]][cities_list[i + 1]]
        distance += self.world_map.distances[cities_list[0]][cities_list[-1]]
        return distance

    def population_fitness(self):
        """Zwraca wartości przystosowania dla całej populacji."""
        distance = np.vectorize(self.route_distance)(self.current_population)
        current_fitness = 1 / distance
        best_index = np.argmax(current_fitness)
        self.current_best_member = self.current_population[best_index]
        self.current_worst_index = np.argmin(current_fitness)
        return current_fitness

    def selection(self):
        """Zwraca tablicę indeksów chromosomów do reprodukcji dla danej populacji."""
        self.current_fitness = self.population_fitness()
        fitness_array = self.current_fitness / np.sum(self.current_fitness)
        roulette_wheel = np.cumsum(fitness_array)
        chosen_chromosomes = np.empty_like(self.current_population, dtype=int)
        for i in range(self.population_size):
            rand = np.random.rand()
            chosen_chromosomes[i] = np.digitize(rand, roulette_wheel)
        return chosen_chromosomes

    def create_pairs(self, selected):
        """Zwraca pary do rozmnożenia dla wybranych chromosomów z danej populacji.. """
        pool = np.copy(selected)
        pairs = np.empty((int(self.population_size / 2), 2), dtype=int)
        for i in range(int(self.population_size / 2)):
            first = np.random.randint(len(pool))
            pairs[i][0] = pool[first]
            pool = np.delete(pool, first)
            second = np.random.choice(len(pool))
            pairs[i][1] = pool[second]
            pool = np.delete(pool, second)
        return pairs

    def crossover(self, pairs):
        """Zwraca nową, rozmnożoną populację dla danej populacji i wybranych z niej indeksów par. """
        new_population = np.empty_like(self.current_population)
        for i, j in zip(range(len(pairs)), range(0, self.population_size, 2)):
            parent1 = self.current_population[pairs[i][0]]
            parent2 = self.current_population[pairs[i][1]]
            pre1, pre2, post1, post2 = [], [], [], []
            cut_points = np.random.choice(range(1, len(parent1)), 2, replace=False)
            cut_point1, cut_point2 = np.min(cut_points), np.max(cut_points)
            middle_substr1 = parent1[cut_point1:cut_point2]
            middle_substr2 = parent2[cut_point1:cut_point2]
            m, n = 0, 0
            while m < len(parent1[:cut_point1]) + len(parent1[cut_point2:]):
                if parent2[n] not in middle_substr1:
                    if m < len(parent1[:cut_point1]):
                        pre1 += parent2[n]
                        m += 1
                    else:
                        post1 += parent2[n]
                        m += 1
                n += 1
            m, n = 0, 0
            while m < len(parent1[:cut_point1]) + len(parent1[cut_point2:]):
                if parent1[n] not in middle_substr2:
                    if m < len(parent1[:cut_point1]):
                        pre2 += parent1[n]
                        m += 1
                    else:
                        post2 += parent1[n]
                        m += 1
                n += 1
            offspring1 = ''.join(pre1 + list(middle_substr1) + post1)
            offspring2 = ''.join(pre2 + list(middle_substr2) + post2)
            new_population[j] = offspring1
            new_population[j + 1] = offspring2
        return new_population

    def mutate(self, population):
        """Zwraca populację po mutacji."""
        mutated = np.copy(population)
        for i in range(self.population_size):
            if np.random.random() < MUTATION_PROBABILITY:
                cut_points = np.random.choice(range(1, self.number_of_cities - 1), 2, replace=False)
                cut_point1, cut_point2 = np.min(cut_points), np.max(cut_points)
                mutated[i] = (population[i][:cut_point1] +
                              population[i][cut_point1:cut_point2][::-1] + population[i][cut_point2:])
        return mutated

    def plot_world(self, gen):
        """Wyświetla wykres aktualnego najlepszego rozwiązania dla danego numeru generacji."""
        best_route = list(self.current_population[np.argmax(self.current_fitness)])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([i for i in range(0, 1001, 100)])
        ax.set_yticks([i for i in range(0, 1001, 100)])
        ax.set_xlim(-100, 1100)
        ax.set_ylim(-100, 1100)
        ax.set_axisbelow(True)
        ax.clear()
        ax.set_title("Działanie algorytmu w czasie rzeczywistym: generacja " + str(gen))
        for i in range(self.number_of_cities - 1):
            x_1 = self.world_map.map["x"][best_route[i]]
            y_1 = self.world_map.map["y"][best_route[i]]
            x_2 = self.world_map.map["x"][best_route[i + 1]]
            y_2 = self.world_map.map["y"][best_route[i + 1]]
            ax.arrow(x_1, y_1, x_2 - x_1, y_2 - y_1, color='r')
        ax.arrow(self.world_map.map["x"][best_route[-1]], self.world_map.map["y"][best_route[-1]],
                 self.world_map.map["x"][best_route[0]] - self.world_map.map["x"][best_route[-1]],
                 self.world_map.map["y"][best_route[0]] - self.world_map.map["y"][best_route[-1]], color='r')
        ax.scatter(self.world_map.map.x, self.world_map.map.y, color='black', s=50)
        for i, txt in enumerate(list(self.world_map.city_pool)):
            plt.annotate(txt, (self.world_map.map.x[i] + 10, self.world_map.map.y[i] + 10))

        fig.canvas.draw()
        plt.pause(0.01)

    def cycle(self, n):
        """Wykonuje n cykli działania algorytmu genetycznego (n generacji). Pokazuje wykres postępu."""
        average_fitness = np.empty(n)
        max_fitness = np.empty(n)
        for g in range(n):
            selected_chromosomes = self.selection()  # Selekcja.
            created_pairs = self.create_pairs(selected_chromosomes)  # Stworzenie par.
            self.current_population = self.mutate(self.crossover(created_pairs))  # Reprodukcja
            self.current_population[self.current_worst_index] = self.current_best_member
            self.current_fitness = self.population_fitness()
            average_fitness[g] = np.mean(self.current_fitness)
            max_fitness[g] = np.max(self.current_fitness)
            if g % 50 == 0:
                gen_alg.plot_world(g)

        plt.figure(figsize=(8, 5))
        plt.xlabel('Generacja')
        plt.ylabel('Funkcja przystosowania')
        plt.plot(range(n), average_fitness, 'blue', label='średnia')
        plt.plot(range(n), max_fitness, 'orange', label='maksimum')
        plt.legend()
        plt.figure(figsize=(8, 5))
        plt.xlabel('Generacja')
        plt.ylabel('Całkowita odległość')
        plt.plot(range(n), 1 / max_fitness, 'blue')
        plt.show()

    def greedy_algorithm(self):
        """Zwraca trasę i jej dystans dla algorytmu zachłannego."""
        city_pool = self.world_map.city_pool[1:]
        greedy_route = "A"
        for i in range(self.number_of_cities - 1):
            closest_city = self.world_map.distances[greedy_route[i]][list(city_pool)].idxmin()
            city_pool = city_pool.replace(closest_city, "")
            greedy_route += closest_city
        return greedy_route, self.route_distance(greedy_route)


ax = fig.add_subplot(1, 1, 1)
gen_alg = GenAlgTSP(NUMBER_OF_CITIES, POPULATION_SIZE, MUTATION_PROBABILITY)
gen_alg.cycle(500)
print("Dystanse:\n", "Algorytm genetyczny:", gen_alg.route_distance(gen_alg.current_best_member),
      "\n Algorytm zachłanny: ", gen_alg.greedy_algorithm()[1])
