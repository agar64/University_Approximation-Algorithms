import numpy as np
from scipy.optimize import linprog
import pulp
import random


def gerar_dados(tamanho_universo=10, num_conjuntos=9, max_elementos_por_conjunto=3, max_custo=10):
    # Gera o universo
    universo = list(range(1, tamanho_universo + 1))

    # Gera conjuntos garantindo que todos os elementos do universo sejam cobertos
    conjuntos = []
    elementos_faltando = set(universo)

    while elementos_faltando and len(conjuntos) < num_conjuntos:
        tamanho_conjunto = random.randint(1, max_elementos_por_conjunto)
        elementos_faltando_lista = list(elementos_faltando)
        conjunto = set(random.sample(universo, min(tamanho_conjunto, len(elementos_faltando_lista))))
        conjuntos.append(conjunto)
        elementos_faltando.difference_update(conjunto)

    # Se ainda houver elementos faltando, cria um conjunto extra para cobri-los
    while len(elementos_faltando) > 0:
        #conjuntos.append(set(elementos_faltando))
        conjunto = set(random.sample(elementos_faltando_lista, min(tamanho_conjunto, len(elementos_faltando_lista))))
        conjuntos.append(conjunto)
        elementos_faltando.difference_update(conjunto)

    # Gera custos aleatórios para cada conjunto
    custos = [random.randint(1, max_custo) for _ in range(len(conjuntos))]

    return universo, conjuntos, custos


def set_cover_exato(U, S, custos):
    # Cria o problema de minimização
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)

    # Variáveis binárias x_i: 1 se o subconjunto S_i for escolhido, 0 caso contrário
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(S))]

    # Função objetivo: minimizar o custo total dos subconjuntos escolhidos
    prob += pulp.lpSum(custos[i] * x[i] for i in range(len(S)))

    # Restrições: cada elemento de U deve ser coberto por pelo menos um subconjunto
    for u in U:
        prob += pulp.lpSum(x[i] for i in range(len(S)) if u in S[i]) >= 1

    # Resolve o problema
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Retorna os subconjuntos escolhidos
    subconjuntos_escolhidos = [i for i in range(len(S)) if pulp.value(x[i]) == 1]
    return subconjuntos_escolhidos

def calcular_f(U, S):
    # Inicializa um dicionário para contar quantas vezes cada elemento de U aparece nos subconjuntos
    frequencias = {u: 0 for u in U}

    # Conta a frequência de cada elemento em U nos subconjuntos
    for S_i in S:
        for u in S_i:
            frequencias[u] += 1

    # Retorna o valor máximo das frequências
    return max(frequencias.values())


def minCC_hochbaum(U, S, custos):

    # Criar a matriz de restrições para o problema de PL
    A = np.zeros((len(U), len(S)))
    for i, u in enumerate(U):
        for j, S_j in enumerate(S):
            if u in S_j:
                A[i, j] = 1

    # Resolver a relaxação linear (minimizar custo sujeita a Ax >= 1)
    result = linprog(custos, A_ub=-A, b_ub=-np.ones(len(U)), bounds=(0, 1))

    # Vetor solução
    x = result.x

    # Calcular f e retornar o conjunto de cobertura
    f = max(np.sum(A, axis=1))  # Frequência máxima de qualquer u
    C = [S_j for j, S_j in enumerate(S) if x[j] >= 1 / f]

    return C


def minCC_hochbaum_dual(U, S, custos):

    # Criar a matriz de restrições
    A = np.zeros((len(U), len(S)))
    for i, u in enumerate(U):
        for j, S_j in enumerate(S):
            if u in S_j:
                A[i, j] = 1

    # Resolver o dual (maximizar y · 1, sujeita a Aᵀ y <= custos)
    result = linprog(-np.ones(len(U)), A_ub=A.T, b_ub=custos, bounds=(0, None))

    # Vetor solução dual
    y = result.x

    # Selecionar o conjunto onde a soma do y[u] de seus elementos é igual ao seu custo
    C = [S_j for j, S_j in enumerate(S) if sum(y[i] for i, u in enumerate(U) if u in S_j) == custos[j]]

    return C


def minCC_primal_dual(U, S, custos):
    # Inicializa o vetor dual y com zeros para todos os elementos de U
    y = {u: 0 for u in U}

    # Inicializa um conjunto para controlar os elementos já cobertos
    cobertos = set()

    # Inicializa o vetor de conjuntos final
    C = []

    # Função auxiliar para verificar se todos os elementos estão cobertos
    def all_elements_covered(U, cobertos):
        return set(U) == cobertos

    # Loop até que todos os elementos estejam cobertos
    while not all_elements_covered(U, cobertos):
        for u in U:
            if u not in cobertos:  # Só atualiza y[u] se o elemento ainda não estiver coberto
                min_c = float('inf')

                # Para cada subconjunto que contém u, calcula quanto falta para cobrir o custo
                for i, S_i in enumerate(S):
                    if u in S_i:
                        soma_y = 0
                        for elem in S_i:
                            soma_y += y[elem]

                            min_c = min(min_c, custos[i] - soma_y) # Atualiza min_c

                y[u] += min_c # Incrementa y[u]

        # Atualiza o conjunto de elementos cobertos
        for i, S_i in enumerate(S):
            soma_y = 0
            for u in S_i:
                soma_y += y[u]
            # Se o subconjunto estiver completamente coberto, adiciona os elementos a 'cobertos'
            if soma_y == custos[i]:
                cobertos.update(S_i)
                if S_i not in C:
                    C.append(S_i)

    return C

# Dados
#U = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#S = [{1, 2, 3}, {3, 4, 5}, {5, 6, 7}, {7, 8, 9}, {9, 10}, {1, 6, 10}, {1, 2}, {1, 10}, {1, 4, 9}]
#custos = [5, 7, 6, 3, 4, 8, 4, 99, 1]
U, S, custos = gerar_dados(tamanho_universo=random.randint(10, 15), num_conjuntos=random.randint(5, 20),
                           max_elementos_por_conjunto=random.randint(3, 5))
print(f"U: {U}")
print(f"S: {S}")
print(f"Custos: {custos}")

subconjuntos = set_cover_exato(U, S, custos)
res = [S[i] for i in subconjuntos]
print(f"Solução Exata: {res} - {len(res)}")

f = calcular_f(U, S)
print(f"Valor de f: {f}")

# Usando os algoritmos
C = minCC_hochbaum(U, S, custos)
print(f"Set Cover (Primal) {f}-aproximativo: {C} - {len(C)}")

C2 = minCC_hochbaum_dual(U, S, custos)
print(f"Set Cover (Dual) {f}-aproximativo: {C2} - {len(C2)}")

C3 = minCC_primal_dual(U, S, custos)
print(f"Set Cover (Primal-Dual) {f}-aproximativo: {C3} - {len(C3)}")