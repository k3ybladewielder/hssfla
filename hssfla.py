import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy  # Para cálculo da entropia
import random
from tqdm import tqdm
from rouge_score import rouge_scorer
import pandas as pd
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")


# Função para inicializar a população
def initialize_population(pop_size, n_sentences):
    """
    Inicializa uma população de vetores binários de tamanho pop_size. Cada vetor representa
    a seleção de sentenças.

    :param pop_size: Tamanho da população
    :param n_sentences: Número total de sentenças (tamanho de cada vetor binário)
    :return: Lista de vetores binários (indivíduos)
    """
    return [np.random.randint(2, size=n_sentences).tolist() for _ in range(pop_size)]

# Função auxiliar para calcular a informatividade (entropia)
def calculate_entropy(individual, alfa):
    """
    Calcula a entropia das sentenças selecionadas em um indivíduo.

    :param individual: Vetor binário indicando as sentenças selecionadas
    :param alfa: Importância relativa das sentenças
    :return: Valor da entropia
    """
    # Garante que `individual` seja convertido em um array NumPy de uma única dimensão
    individual = np.ravel(np.array(individual))

    # Filtra os valores de alfa correspondentes às sentenças selecionadas
    alfa_selected = alfa[individual == 1]
    if alfa_selected.size == 0:
        return 0  # Caso nenhuma sentença esteja selecionada, a entropia é zero

    # Normaliza os valores de alfa selecionados para somar 1 (distribuição de probabilidade)
    p = alfa_selected / np.sum(alfa_selected)

    # Calcula a entropia usando a fórmula de Shannon
    entropia = -np.sum(p * np.log2(p))
    return entropia


def calculate_objective_function(x, alfa, sim, L, li, beta, gamma, epsilon=1e-8):
    """
    Calcula o valor da função objetivo para uma população de resumos.

    :param x: População (lista de vetores binários)
    :param alfa: Vetor de relevâncias (\( \alpha \))
    :param sim: Matriz de similaridade entre sentenças
    :param L: Limite de comprimento do resumo
    :param li: Tamanhos das sentenças
    :param beta: Parâmetro de penalização de comprimento (para o fitness, não aqui)
    :param gamma: Peso da informatividade (entropia)
    :param epsilon: Pequena constante para evitar divisão por zero
    :return: Vetor contendo o valor da função objetivo para cada indivíduo na população
    """
    # Converter x para um array NumPy, caso não seja
    x = np.array(x)

    # Número de soluções
    N = x.shape[0]

    # Inicializa o vetor da função objetivo
    objective_values = np.zeros(N)

    # Calcula a relevância total para cada solução
    relevance_scores = np.dot(x, alfa)

    # Inicializa a penalização de redundância (inversa da soma das similaridades)
    redundancy_penalty_inverse = np.zeros(N)

    # Calcula a soma das similaridades entre pares de sentenças selecionadas
    for k in range(N):  # Para cada indivíduo na população
        sum_similarities = 0
        selected_indices = np.where(x[k] == 1)[0]
        if len(selected_indices) > 1:
            for i_idx in range(len(selected_indices)):
                for j_idx in range(i_idx + 1, len(selected_indices)):
                    i = selected_indices[i_idx]
                    j = selected_indices[j_idx]
                    sum_similarities += sim[i, j]
        redundancy_penalty_inverse[k] = 1 / (sum_similarities + epsilon) if sum_similarities >= 0 else 1 / epsilon

    # Calcula a entropia (informatividade) para cada indivíduo
    informativity_scores = np.array([calculate_entropy(individual, alfa) for individual in x])

    # Calcula o valor da função objetivo
    objective_values = relevance_scores * redundancy_penalty_inverse + gamma * informativity_scores

    return objective_values

# Ajuste da função para cálculo individual da função objetivo
def calculate_objective_function_individual(x, alfa, sim, L, li, beta, gamma, epsilon=1e-8):
    """
    Calcula o valor da função objetivo para um único indivíduo (resumo).

    :param x: Indivíduo (vetor binário)
    :param alfa: Vetor de relevâncias (\( \alpha \))
    :param sim: Matriz de similaridade entre sentenças
    :param L: Limite de comprimento do resumo
    :param li: Tamanhos das sentenças
    :param beta: Parâmetro de penalização de comprimento (para o fitness, não aqui)
    :param gamma: Peso da informatividade (entropia)
    :param epsilon: Pequena constante para evitar divisão por zero
    :return: Valor da função objetivo para o indivíduo
    """
    # Converter x para um array NumPy, caso não seja
    x = np.array(x)

    # Calcula a relevância total para a solução individual
    relevance_score = np.dot(x, alfa)

    # Inicializa a soma das similaridades
    sum_similarities = 0
    selected_indices = np.where(x == 1)[0]
    if len(selected_indices) > 1:
        for i_idx in range(len(selected_indices)):
            for j_idx in range(i_idx + 1, len(selected_indices)):
                i = selected_indices[i_idx]
                j = selected_indices[j_idx]
                sum_similarities += sim[i, j]

    # Calcula a penalização de redundância (inversa da soma das similaridades)
    redundancy_penalty_inverse = 1 / (sum_similarities + epsilon) if sum_similarities >= 0 else 1 / epsilon

    # Calcula a entropia (informatividade)
    informativity_score = calculate_entropy(x, alfa)

    # Calcula o valor da função objetivo
    objective_value = relevance_score * redundancy_penalty_inverse + gamma * informativity_score

    return objective_value


def calculate_fitness(x, alfa, sim, L, li, beta, gamma, epsilon=1e-8):
    """
    Calcula o fitness para uma população considerando a função objetivo e a penalidade de comprimento.

    :param x: População (lista de vetores binários)
    :param alfa: Vetor de relevâncias (\( \alpha \))
    :param sim: Matriz de similaridade entre sentenças
    :param L: Limite de comprimento do resumo
    :param li: Tamanhos das sentenças
    :param beta: Parâmetro de penalização de comprimento
    :param gamma: Peso da informatividade (entropia)
    :param epsilon: Pequena constante para evitar divisão por zero
    :return: Vetor de fitness para a população
    """
    # Calcula o valor da função objetivo
    objective_values = calculate_objective_function(x, alfa, sim, L, li, beta, gamma, epsilon)

    # Calcula a penalização por comprimento
    length_penalty = np.maximum(0, np.dot(x, li) - L)

    # Aplica a penalização ao valor da função objetivo para obter o fitness
    fitness = objective_values * np.exp(-beta * (length_penalty ** 2))

    return fitness

# Ajuste da função para cálculo individual de fitness
def calculate_fitness_individual(x, alfa, sim, L, li, beta, gamma, epsilon=1e-8):
    """
    Calcula o fitness para um único indivíduo (resumo) considerando a função objetivo
    e a penalidade de comprimento.

    :param x: Indivíduo (vetor binário)
    :param alfa: Vetor de relevâncias (\( \alpha \))
    :param sim: Matriz de similaridade entre sentenças
    :param L: Limite de comprimento do resumo
    :param li: Tamanhos das sentenças
    :param beta: Parâmetro de penalização de comprimento
    :param gamma: Peso da informatividade (entropia)
    :param epsilon: Pequena constante para evitar divisão por zero
    :return: Fitness do indivíduo
    """
    # Calcula o valor da função objetivo para o indivíduo
    objective_value = calculate_objective_function_individual(x, alfa, sim, L, li, beta, gamma, epsilon)

    # Calcula a penalização por comprimento
    length_penalty = max(0, np.dot(x, li) - L)

    # Aplica a penalização ao valor da função objetivo para obter o fitness
    fitness = objective_value * np.exp(-beta * (length_penalty ** 2))

    return fitness


# Função para calcular a similaridade média
def calculate_average_similarity(individual, similarities):
    """
    Calcula a similaridade média das frases presentes no indivíduo com o centroide.

    :param individual: Vetor binário indicando as sentenças selecionadas
    :param similarities: Vetor de similaridades das sentenças com o centroide
    :return: Similaridade média
    """
    # Filtra as similaridades com base no indivíduo
    similarities_atual = similarities[individual == 1]
    return np.mean(similarities_atual) if similarities_atual.size > 0 else 0

# Função para selecionar uma frase aleatória ausente
def select_random_absent_sentence(individual):
    """
    Seleciona o índice de uma frase que não está presente no indivíduo aleatoriamente.

    :param individual: Vetor binário indicando as sentenças selecionadas
    :return: Índice da frase ausente ou None se todas as frases estiverem presentes
    """
    # Encontra os índices das frases que não estão no indivíduo
    absent_sentences_indices = np.where(individual == 0)[0]
    if absent_sentences_indices.size > 0:
        return random.choice(absent_sentences_indices)
    return None

# Mutação
def mutate(individual, similarities_to_centroid, centroid):
    """
    Aplica uma operação de mutação (adicionar, remover ou trocar) a um indivíduo.
    A probabilidade de cada operação é igual.

    :param individual: Vetor binário representando um indivíduo
    :param similarities_to_centroid: Similaridade de cada frase com o centroide
    :param centroid: Vetor do centroide do documento
    :return: Novo indivíduo mutado
    """
    mutation_type = random.choice(['add', 'remove', 'swap'])
    #mutated_individual = individual.copy()
    mutated_individual = np.array(individual).flatten()

    if mutation_type == 'add':
        # Adicionar uma frase
        average_similarity = calculate_average_similarity(mutated_individual, similarities_to_centroid)

        # Tenta encontrar uma frase ausente com similaridade maior que a média
        zero_indices = np.where(mutated_individual == 0)[0]
        potential_additions = [idx for idx in zero_indices if similarities_to_centroid[idx] > average_similarity]

        if potential_additions:
            idx_to_add = random.choice(potential_additions)
        else:
            # Se nenhuma frase satisfaz a condição, escolhe aleatoriamente uma ausente
            idx_to_add = select_random_absent_sentence(mutated_individual)

        if idx_to_add is not None:
            mutated_individual[idx_to_add] = 1

    elif mutation_type == 'remove':
        # Remover uma frase
        average_similarity = calculate_average_similarity(mutated_individual, similarities_to_centroid)

        one_indices = np.where(mutated_individual == 1)[0]
        removable_sentences = [idx for idx in one_indices if similarities_to_centroid[idx] < average_similarity]

        if removable_sentences:
            idx_to_remove = random.choice(removable_sentences)
            mutated_individual[idx_to_remove] = 0

    elif mutation_type == 'swap':
        # Trocar uma frase presente por uma ausente
        zero_indices = np.where(mutated_individual == 0)[0]
        one_indices = np.where(mutated_individual == 1)[0]

        if zero_indices.size > 0 and one_indices.size > 0:
            idx_to_remove = random.choice(one_indices)
            idx_to_add = random.choice(zero_indices)
            mutated_individual[idx_to_remove] = 0
            mutated_individual[idx_to_add] = 1

    return mutated_individual


def calculate_L(sentences_list, factor=0.5):
    """
    Calcula L como uma fração do somatório dos tamanhos das sentenças na lista.

    :param sentences_list: Lista de sentenças
    :param fator: Fator para calcular a fração do somatório (0.5 para metade, 1/3 para um terço, etc.)
    :return: Valor calculado para L
    """
    # Calcula o somatório dos tamanhos das sentenças
    somatorio_tamanhos = sum(len(sentence) for sentence in sentences_list)

    # Calcula L com base no fator fornecido
    L = fator * somatorio_tamanhos

    return L


def hssfla(embeddings,
                      preprocessed_sentences_list,
                      n_cycle=10,
                      n_iter=100,
                      n_memeplex=5,
                      beta_max=0.5,  # Valor final de beta
                      beta_min=0.1,  # Valor inicial de beta
                      L=1000,
                      pop_size=10,
                      gamma=0.5,
                      epsilon=1e-8):
    """
    Executa o pipeline do algoritmo MRMRSFLA para otimização da função objetivo de resumo de texto.

    :param embeddings: Matriz de embeddings das sentenças.
    :param preprocessed_sentences_list: Lista de sentenças pré-processadas.
    :param n_cycle: Número de ciclos do algoritmo.
    :param n_iter: Número de iterações de melhoria local por ciclo.
    :param n_memeplex: Número de memeplexes (subgrupos da população).
    :param beta_max: Valor máximo do parâmetro de penalização de comprimento.
    :param beta_min: Valor mínimo do parâmetro de penalização de comprimento.
    :param L: Limite de comprimento do resumo.
    :param pop_size: Tamanho da população.
    :param gamma: Peso do termo de entropia na função objetivo.
    :param epsilon: Pequena constante para evitar divisão por zero na função objetivo.
    :return: Uma tupla contendo:
              - best_global_individual: O melhor indivíduo (vetor binário) encontrado.
              - best_global_fitness: O fitness do melhor indivíduo.
              - best_sentences: A lista das sentenças correspondentes ao melhor indivíduo.
    """
    n_sentences = embeddings.shape[0]  # Número total de sentenças
    population = initialize_population(pop_size, n_sentences)  # Inicializa a população

    # Calcula a matriz de similaridade entre as frases
    similarity_matrix = cosine_similarity(embeddings)

    # Calcula o centroide (média dos embeddings)
    centroid = np.mean(embeddings, axis=0)

    # Calcula a similaridade de cada frase com o centroide
    similarity_to_centroid = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()

    # Calcula o vetor alfa (relevância das sentenças)
    sum_similarity_centroid = np.sum(similarity_to_centroid)
    alfa = similarity_to_centroid / sum_similarity_centroid if sum_similarity_centroid > 0 else np.ones(n_sentences) / n_sentences

    # Tamanho das sentenças
    sentence_lengths = np.array([len(sentence) for sentence in preprocessed_sentences_list])

    # Inicializa o beta com o valor mínimo
    beta = beta_min

    # Calcula o fitness inicial da população
    fitness = calculate_fitness(population, alfa, similarity_matrix, L, sentence_lengths, beta, gamma, epsilon)

    # Ordena a população e o fitness com base no fitness (maior primeiro)
    sorted_fitness_indices = np.argsort(fitness)[::-1]
    sorted_fitness = fitness[sorted_fitness_indices]
    sorted_population = [population[i] for i in sorted_fitness_indices]

    # Inicializa o melhor indivíduo e fitness global
    best_global_fitness = sorted_fitness[0]
    best_global_individual = sorted_population[0]

    # Ciclos de evolução
    for cycle in tqdm(range(n_cycle), desc="Processing HSSFLA: "):
        # Dividir a população em memeplexes
        memeplexes = [sorted_population[i::n_memeplex] for i in range(n_memeplex)]
        fitness_memeplexes = [sorted_fitness[i::n_memeplex] for i in range(n_memeplex)]

        updated_population = []

        # Iterações de melhoria de cada memeplexe
        for iteration in range(n_iter):
            # Atualiza dinamicamente o beta
            beta = beta_min + (beta_max - beta_min) * (iteration / n_iter)

            # Para cada memeplexe
            for memeplex_idx, (memeplex, fitness_m) in enumerate(zip(memeplexes, fitness_memeplexes)):
                # Melhor e pior indivíduo do memeplexe
                best_local_individual = memeplex[0]
                best_local_fitness = fitness_m[0]
                worst_local_individual = memeplex[-1]
                worst_local_fitness = fitness_m[-1]

                # Mutação do melhor indivíduo local
                mutated_best_local = mutate(best_local_individual, similarity_to_centroid, centroid)
                fitness_mutated_best_local = calculate_fitness_individual(mutated_best_local, alfa, similarity_matrix, L, sentence_lengths, beta, gamma, epsilon)

                if fitness_mutated_best_local > worst_local_fitness:
                    # Substitui o pior indivíduo do memeplexe pelo mutante local
                    memeplexes[memeplex_idx][-1] = mutated_best_local
                    fitness_memeplexes[memeplex_idx][-1] = fitness_mutated_best_local
                    # Reordena o memeplexe
                    sorted_indices_local = np.argsort(fitness_memeplexes[memeplex_idx])[::-1]
                    memeplexes[memeplex_idx] = [memeplexes[memeplex_idx][i] for i in sorted_indices_local]
                    fitness_memeplexes[memeplex_idx] = [fitness_memeplexes[memeplex_idx][i] for i in sorted_indices_local]
                else:
                    # Mutação do melhor indivíduo global
                    mutated_best_global = mutate(best_global_individual, similarity_to_centroid, centroid)
                    fitness_mutated_best_global = calculate_fitness_individual(mutated_best_global, alfa, similarity_matrix, L, sentence_lengths, beta, gamma, epsilon)

                    if fitness_mutated_best_global > worst_local_fitness:
                        # Substitui o pior indivíduo do memeplexe pelo mutante global
                        memeplexes[memeplex_idx][-1] = mutated_best_global
                        fitness_memeplexes[memeplex_idx][-1] = fitness_mutated_best_global
                        # Reordena o memeplexe
                        sorted_indices_local = np.argsort(fitness_memeplexes[memeplex_idx])[::-1]
                        memeplexes[memeplex_idx] = [memeplexes[memeplex_idx][i] for i in sorted_indices_local]
                        fitness_memeplexes[memeplex_idx] = [fitness_memeplexes[memeplex_idx][i] for i in sorted_indices_local]
                    else:
                        # Substitui o pior indivíduo por um novo indivíduo aleatório
                        random_individual = initialize_population(1, n_sentences)[0]
                        memeplexes[memeplex_idx][-1] = random_individual
                        fitness_memeplexes[memeplex_idx][-1] = calculate_fitness_individual(random_individual, alfa, similarity_matrix, L, sentence_lengths, beta, gamma, epsilon)
                        # Reordena o memeplexe
                        sorted_indices_local = np.argsort(fitness_memeplexes[memeplex_idx])[::-1]
                        memeplexes[memeplex_idx] = [memeplexes[memeplex_idx][i] for i in sorted_indices_local]
                        fitness_memeplexes[memeplex_idx] = [fitness_memeplexes[memeplex_idx][i] for i in sorted_indices_local]

        # Reconstroi a população a partir dos memeplexes
        updated_population = [item for sublist in memeplexes for item in sublist]
        fitness = np.array([calculate_fitness_individual(ind, alfa, similarity_matrix, L, sentence_lengths, beta, gamma, epsilon) for ind in updated_population])

        # Ordena a população atualizada
        sorted_fitness_indices = np.argsort(fitness)[::-1]
        sorted_fitness = fitness[sorted_fitness_indices]
        sorted_population = [updated_population[i] for i in sorted_fitness_indices]

        # Seleciona os 'pop_size' melhores sobreviventes para o próximo ciclo
        sorted_population = sorted_population[:pop_size]
        sorted_fitness = sorted_fitness[:pop_size]

        # Atualiza o melhor indivíduo e fitness global
        if sorted_fitness[0] > best_global_fitness:
            best_global_fitness = sorted_fitness[0]
            best_global_individual = sorted_population[0]

    # Retornar o melhor indivíduo, o melhor fitness e as frases do melhor indivíduo
    best_sentences = [preprocessed_sentences_list[i] for i, selected in enumerate(best_global_individual) if selected == 1]

    return best_global_individual, best_global_fitness, best_sentences
