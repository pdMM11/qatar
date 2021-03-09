# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
from operator import itemgetter
from random import randint, random, shuffle, choice
from copy import copy

class Qatar:

    def __init__(self, path, dataset = None, random_init = False, remove_meta_data = False,
                 islands = 10, island_size = 10, niter = 50000, niter_migrate = 20, tam_quasi_greedy = 3):
        """
        :param path: path para o ficheiro contendo as cidades e as distâncias
        :param dataset: alternativamente, pode-se fazer load de um dataset já inicializado, que permite passar à frente o passo inicial quasi greedy
        :param random_init: define se é executada uma inicialização aleatória ou quasi greedy
        :param remove_meta_data: define se os metadados do ficheiro inicial de distâncias serão removidos ou não
        :param islands: nº de ilhas criadas
        :param island_size: nº de indivíduos em cada ilha
        :param niter: nº iterações até ao algoritmo acabar
        :param niter_migrate: de quantas em quantas iterações ocorre troca de indivíduos entre ilhas
        :param tam_quasi_greedy: nº de cidades mais próximas de uma dada a ser usadas para o método quasi greedy
        """

        os.chdir(path)
        self.islands = islands
        self.island_size = island_size
        self.niter = niter
        self.niter_migrate = niter_migrate
        self.tam_quasi_greedy = tam_quasi_greedy
        self.record = []  # verifica a melhor solução
        if remove_meta_data:
            try: remove_meta(path)
            except: print("Erro na limpeza dos metadados")
        try: self.qatar = pd.read_csv("qatar_file.txt", delimiter=" ", header = None, index_col = 0)
        except:
            print("Erro: Na leitura de dados")
            self.qatar = None

        if self.qatar is not None:
            self.n_total = self.qatar.shape[0]  # nº de cidades
            self.dists = self.distance_matrix()
            if dataset is None:
                if not random_init: self.popul = self.create_islands()
                else: self.popul = self.random_init()
            else: self.popul = dataset
            self.init_popul = copy(self.popul)
            self.fitness_all = self.fitness_indivs()
            # print(self.fitness_all[0])



    def distance_matrix(self):
        # Este array, que é uma matriz triangular superior, so terá as distancias da cidade com index menor para a cidade de index maior
        # Assim evitam-se dados cálculos de distância com o próprio e cálculos repetidos, pois a direção inversa é igual
        dists = []
        for i in range(1,len(self.qatar)):
            for j in range(i+1,len(self.qatar)+1):
                dist_ij = (((self.qatar.loc[i,1] - self.qatar.loc[j,1])**2
                            + (self.qatar.loc[i,2] - self.qatar.loc[j,2])**2))**(0.5)
                dists.append([i,j,dist_ij])

        return np.asarray(dists)

    def index_distance(self, x1, x2):
        # Através dos indexes das 2 cidades, vê qual o maior e o menor e calcula o index em que se encontra o valor da
        # distância na matriz de distâncias. Retorna um array contendo os indexes e a distância entre essas cidades.
        menor = min(x1,x2)
        maior = max(x1,x2)
        index = int((menor-1)*(self.n_total) + maior - ((1+menor)/2)*(menor) - 1)
        return self.dists[index]


    def random_init(self):
        # Inicialização completamente aleatória
        islands = []
        while len(islands) < self.islands:
            islands_i = []
            while len(islands_i) < self.island_size:
                islands_i.append(np.random.choice(range(1,self.n_total+1), self.n_total, replace=False))
            islands.append(islands_i)
        return np.asarray(islands)

    def nearest_cities_i(self, city, already_city = []):
        # Dada uma cidade, vai retornar uma lista com as 3 cidades mais próximas (valor default definido por tam_quasi_greedy)
        # Nos últimos casos, em que já não é possível devolver 3 cidades, porque já foram todas utilizadas, retorna uma lista menor
        dists_city = []
        tam_inicial = self.tam_quasi_greedy
        tam = tam_inicial
        for dist in self.dists:
            if dist[0] == city:
                if dist[1] not in already_city:
                    dists_city.append(dist)
            elif dist[1] == city:
                if dist[0] not in already_city:
                    dists_city.append(dist)
        best_dists = sorted(dists_city, key=itemgetter(2))
        if len(already_city) > self.n_total - tam_inicial:
            tam = self.n_total - len(already_city)
        best_dists = best_dists[:tam]
        choices = []
        for dist in best_dists:
            if dist[0] == city: choices.append(dist[1])
            else: choices.append(dist[0])
        return choices


    def create_islands(self):
        # Inicialização quasy greedy. Retorna as ilhas criadas
        # Tem uma inicialização completamente aleatória para cada individuo de cada ilha e depois procura as cidades mais próximas (nearest_cities_i)
        islands = []
        while len(islands) < self.islands:
            islands_i = []
            while len(islands_i) < self.island_size:
                already_city = []
                i = randint(1, self.n_total)
                already_city.append(i)
                while len(already_city) < self.n_total:
                    choices = self.nearest_cities_i(i, already_city)
                    i = choice(choices)
                    already_city.append(i)
                islands_i.append(already_city)
                print(len(islands), " ", len(islands_i))
            islands.append(islands_i)
        return np.asarray(islands)


    def fitness(self, sol):
        # Calcula o fitness para um dado indivíduo.
        # Começa por calcular a distância entre a 1ª e última cidade e depois adiciona as restantes distâncias.
        # Retorna a distância total
        fitness = self.index_distance(sol[0], sol[-1])[2]
        for i in range(len(sol)-1): fitness += self.index_distance(sol[i],sol[i+1])[2]
        return fitness

    def fitness_indivs(self):
        # Para cada indivíduo de cada ilha calcula o fitness.
        # Retorna um dicionário contento os valores para todos os indivíduos de todas as ilhas
        fitness = {}
        index_i = 0
        for i in self.popul:
            fitness_i = {}
            index_j = 0
            for j in i:
                fitness_i[index_j] = self.fitness(j)
                index_j += 1
            fitness[index_i] = fitness_i
            index_i += 1
        return fitness

    def best_worst_fitness(self, dictionary, how_many = 4):
        # Retorna as 4 (default) melhores e piores fitnesses em listas
        res_best = sorted(dictionary.items(), key = itemgetter(1))[:how_many]
        res_worst = sorted(dictionary.items(), key = itemgetter(1), reverse=True)[:how_many]
        return (res_best, res_worst)


    def iterDA(self, index, how_many = 4):
        # Vai buscar os 4 (default) melhores e piores indivíduos de uma ilha e realiza um tipo de mutação em cada um dos
        # melhores. Caso o indivíduo mutado seja melhor que um dos piores, este é substituído pelo indivíduo mutado.
        (best_fit, worst_fit) = self.best_worst_fitness(self.fitness_all[index], how_many)
        for i in range(how_many):
            indiv_best = copy(self.popul[index][best_fit[i][0]])
            type_mutation = np.random.choice([1,2,3],p=[0.4,0.3,0.3])
            mutation = []
            if type_mutation == 1:
                mutation = self.mut2SWAP(indiv_best)
            elif type_mutation == 2:
                mutation = self.mut3SWAP(indiv_best)
            else:
                mutation = self.mutInv(indiv_best)

            if self.fitness(mutation) <= self.fitness_all[index][worst_fit[i][0]] and not checkIfDuplicates(mutation): # erro comum de duplicar cidades; quando descobrirmos o motivo do erro, retirar este método
                self.popul[index][worst_fit[i][0]] = copy(mutation)
                self.fitness_all[index][worst_fit[i][0]] = self.fitness(self.popul[index][worst_fit[i][0]])


    def permute_islands(self):
        # Vai buscar o pior e melhor indivíduo de ilhas adjacentes, realiza mutações no melhor indivíduo e tenta
        # substituir o pior da ilha seguinte pelo indivíduo mutado.
        for i in range(self.islands):
            next_i = i+1
            if next_i == self.islands: next_i = 0
            (best_fit_this, worst_fit_this) = self.best_worst_fitness(self.fitness_all[i],1)
            (best_fit_next, worst_fit_next) = self.best_worst_fitness(self.fitness_all[next_i],1)
            indiv_best = copy(self.popul[i][best_fit_this[0][0]])
            type_mutation = np.random.choice([1,2,3],p=[0.4,0.3,0.3])
            mutation = []
            if type_mutation==1:
                mutation = self.mut2SWAP(indiv_best)
            elif type_mutation==2:
                mutation = self.mut3SWAP(indiv_best)
            else:
                mutation = self.mutInv(indiv_best)

            if self.fitness(mutation) <= self.fitness_all[next_i][worst_fit_next[0][0]] and not checkIfDuplicates(mutation): # erro comum de duplicar cidades; quando descobrirmos o motivo do erro, retirar este método
                self.popul[next_i][worst_fit_next[0][0]] = copy(mutation)
                self.fitness_all[next_i][worst_fit_next[0][0]] = self.fitness(self.popul[next_i][worst_fit_next[0][0]])


    def cycle(self):
        # Para o nº de iterações definidos, irá realizar mutações (iterDA) à população e migrações entre as ilhas, nos
        # intervalos definidos. No fim encontra o melhor indivíduo (com menor fitness) e imprime-o.
        for i in range(self.niter):
            for j in range(self.islands):
                self.iterDA(j)
            if i % self.niter_migrate == 0:
                self.permute_islands()
                print(self.best_current_fit())
                if self.check_local_minimun(): self.increase_diversity(i)

        best_fit = self.best_current_fit()
        print("Best solution:", self.popul[best_fit[0]][best_fit[1]])
        print("Fitness: ", best_fit[2])

    def mut2SWAP(self, individuo):
        # 2 swap mutation - troca duas cidades de posição
        i = 0
        j = 0
        while i == j:
            i = randint(0,self.n_total-1)
            j = randint(0,self.n_total-1)
        individuo[i], individuo[j] = copy(individuo[j]), copy(individuo[i])
        return individuo

    def mut3SWAP(self, individuo):
        # 3 swap mutation - troca 3 cidades de posição entre elas
        i = 0
        j = 0
        k = 0
        while i == j and j == k and i == k:
            i = randint(0, self.n_total-1)
            j = randint(0, self.n_total-1)
            k = randint(0, self.n_total-1)
        i_new, j_new, k_new = 0, 0, 0
        i_new = choice([j, k])
        if i_new == j:
            j_new = k
            k_new = i
        else:
            j_new = i
            k_new = j
        individuo[i], individuo[j], individuo[k] = copy(individuo[i_new]), copy(individuo[j_new]), copy(individuo[k_new])
        return individuo

    def mutInv(self, individuo):
        # Realiza a inversão da ordem das cidades entre um intervalo
        i = 0
        j = 0
        while i == j:
            i = randint(0, self.n_total-1)
            j = randint(0, self.n_total-1)
        individuo[min(i, j):max(i, j)] = copy(individuo[min(i, j):max(i, j)][::-1])
        return individuo


    def increase_diversity(self,i):
        print("Inscreasing diversity at iteraction ", i)
        island_w_best_fit = self.record[len(self.record)-1][0]
        for island in range(len(self.popul)):
            if island != island_w_best_fit:
                self.popul[island] = self.init_popul[island]
        self.fitness_all = self.fitness_indivs()
        self.record = []


    def best_current_fit(self):
        # Procura e retorna o melhor indivíduo, o seu fitness e a ilha em que está
        best_island = 0
        best_indiv = 0
        best_fit = np.inf
        for island in range(len(self.fitness_all)):
            for indiv in range(len(self.fitness_all[island])):
                if self.fitness_all[island][indiv] < best_fit:
                    best_island,best_indiv,best_fit = island,indiv,self.fitness_all[island][indiv]
        self.record.append((best_island,best_indiv,best_fit))
        return (best_island,best_indiv,best_fit)


    def export_population(self):
        return self.popul

    def check_local_minimun(self):
        # Verifica se as soluções estão a convergir para um mínimo local
        if len(self.record) < 50: return False
        return all(ele == self.record[len(self.record)-50] for ele in self.record[len(self.record)-49:])

def checkIfDuplicates(listOfElems):
    # Para combater erro de duplicacao de cidades
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False

#def iterGA(pop):
#    p1,p2,f1,f2 = torneio(pop)
#    ind1, ind2 = pop[p1], pop[p2]
#    pop[f1], pop[f2] = mut(ind1), mut(ind2)


#def mut2SWAP(ind): # Tentar fazer isto
#   while I == J:
#       I = select_random_pop(ind)
#       J = select_random_pop(ind)  
#   nind[I], nind[J] = nind[I], nind[I]
#   fitness -= dist(I-1,I) + dist(I,I+1) + dist(J-1,J) + dist(J,J+1)
#   fitness += dist(I-1,J) + dist(J,I+1) + dist(J-1,I) + dist(I,J+1)
#   nind[I],nind[J]=nind[J],nind[I]


def remove_meta(path):
    # Remoção dos metadados: remove as primeiras 7 linhas, bem como remover a última linha que não é numérica
    # Cria um novo ficheiro com os dados tratados
    os.chdir(path)
    try:
        qatar = open("qa194.tsp.txt")
        data = qatar.readlines()

        f = open("qatar_file.txt", "w")
        for line in data[7:len(data)-1]: f.write(line)

        f.close()
        qatar.close()

    except:
        print("Erro na limpeza de metadados")


if __name__ == "__main__":
    filepath = "C:\\Users\\Pedro\\OneDrive\\Documentos\\UMinho\\Sistemas Inteligentes\\Trabalho Qatar"
    problem = Qatar(filepath, islands = 10)
    problem.cycle()
    # print(problem.nearest_cities_i(177,[181]))
