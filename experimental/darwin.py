import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_coordinates_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[['X', 'Y']].values

def calculate_distances(coordinates):
    return np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

def total_distance(route, distances):
    return np.sum(distances[route[:-1], route[1:]]) + distances[route[-1], route[0]]

def initialize_population(population_size, num_cities):
    return [np.random.permutation(np.arange(num_cities)) for _ in range(population_size)]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], np.setdiff1d(parent2, parent1[:crossover_point])])
    return child

def mutate(route):
    mutation_point1, mutation_point2 = np.random.choice(len(route), size=2, replace=False)
    route[mutation_point1], route[mutation_point2] = route[mutation_point2], route[mutation_point1]
    return route

def select_parents(population, distances):
    fitness_scores = [1 / (total_distance(route, distances) + 1e-6) for route in population]
    probabilities = np.array(fitness_scores) / np.sum(fitness_scores)
    parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return population[parents_indices[0]], population[parents_indices[1]]

def genetic_algorithm(num_cities, population_size=100, generations=1000):
    city_coordinates = load_coordinates_from_csv('/Users/igormol/Desktop/city_coordinates.csv')  # Update with the actual path
    distances = calculate_distances(city_coordinates)

    population = initialize_population(population_size, num_cities)

    for generation in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, distances)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)

            if np.random.rand() < 0.1:
                child1 = mutate(child1)
            if np.random.rand() < 0.1:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    best_route = min(population, key=lambda route: total_distance(route, distances))
    best_distance = total_distance(best_route, distances)

    return best_route, best_distance

def plot_route(city_coordinates, route, title="TSP Solution"):
    plt.figure(figsize=(8, 6))
    plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], label='Cities', c='red')
    plt.plot(city_coordinates[route, 0], city_coordinates[route, 1], linestyle='-', linewidth=2, label='Route')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    np.random.seed(42)

    num_cities = 10  # Adjust the number of cities as needed
    best_route, best_distance = genetic_algorithm(num_cities)

    print(f"Best Route: {best_route}")
    print(f"Total Distance: {best_distance:.2f}")

    city_coordinates = load_coordinates_from_csv('/Users/igormol/Desktop/city_coordinates.csv')  # Update with the actual path
    plot_route(city_coordinates, best_route, title="TSP Solution - Genetic Algorithm")

if __name__ == "__main__":
    main()
