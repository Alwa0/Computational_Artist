from PIL import Image
import random
import time
import numpy as np
start_time = time.time()

frac = 10
initial_population = 70
mutation_rate = 0.95
image = Image.open("steach.jpg")


class Individual:
    def __init__(self):
        self.fitness_score = 0
        self.pixels = np.array([])

    def calculate_fitness(self):
        self.fitness_score = np.abs(self.pixels.astype(np.int) - initial_pixels.astype(np.int)).sum()
        '''
        score = 0
        for r in range(h):
            for p in range(w):
                if np.all(self.pixels[r][p] == initial_pixels[r][p]):
                    score += 1
        self.fitness_score = score'''

    def show(self):
        img = Image.fromarray(self.pixels)
        img.save('./outputs/out.png')


class Population:
    def __init__(self, population_size, initial_image):
        self.image = initial_image
        self.size = population_size
        self.individuals = []

    def create_population(self):
        for ind in range(initial_population):
            pixs = np.array(Image.new("RGB", (w, h), 0))
            for r in range(h):
                for p in range(w):
                    pixs[r][p] = (list(random.choice(colors)))
            individual = Individual()
            individual.pixels = pixs
            individual.calculate_fitness()
            self.individuals.append(individual)

    def sort(self):
        self.individuals.sort(key=lambda ind: ind.fitness_score)

    def offsprings(self):
        best = self.individuals[0:int(initial_population/2)]
        for ind in best:
            parent1 = ind
            parent2 = random.choice(best)
            child = Individual()
            pixs = np.array(Image.new("RGB", (w, h), 0))
            for r in range(h):
                for p in range(w):
                    pixs[r][p] = random.choice([parent1.pixels[r][p], parent2.pixels[r][p]])
                    pixs[r][p] = mutate(pixs[r][p])
            child.pixels = pixs
            child.calculate_fitness()
            self.individuals.append(child)

    def die(self):
        self.individuals = self.individuals[0:initial_population]


def mutate(pixel):
    if random.randint(0, 100) > mutation_rate * 100:
        pixel = list(random.choice(colors))
    return pixel


w = int(image.width/frac)
h = int(image.height/frac)
image = image.resize((w, h)).convert("RGB")
image.save('./outputs/squeezed.png', image.format)
initial_pixels = np.array(image)

colors = Image.Image.getcolors(image, h*w)
colors = [a[1] for a in colors]

population = Population(initial_population, image)
population.create_population()
population.sort()

repetitions = 0
while repetitions < 10000000000:
    repetitions += 1
    population.offsprings()
    population.sort()
    population.die()
    if repetitions % 10 == 0:
        population.individuals[0].show()
        print(population.individuals[0].fitness_score)
population.individuals[0].show()
print(population.individuals[0].fitness_score)
print("--- %s seconds ---" % (time.time() - start_time))
