import time

import numpy as np
import networkx as nx
from scipy.optimize import linprog
import random
import pulp
import matplotlib.pyplot as plt

def vertex_cover_exato(G, custos):
    # Cria o problema de minimização
    prob = pulp.LpProblem("VertexCover", pulp.LpMinimize)

    # Variáveis binárias x_v: 1 se o vértice v for escolhido, 0 caso contrário
    x = {v: pulp.LpVariable(f"x{v}", cat="Binary") for v in G.nodes()}

    # Função objetivo: minimizar o custo total dos vértices escolhidos
    prob += pulp.lpSum(custos[v] * x[v] for v in G.nodes())

    # Restrições: cada aresta (u, v) deve ser coberta por pelo menos um vértice
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1

    # Resolve o problema sem exibir mensagens
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Retorna os vértices escolhidos
    vertices_escolhidos = [v for v in G.nodes() if pulp.value(x[v]) == 1]
    return vertices_escolhidos

def minCV_hochbaum(G, custos):
    n = len(G.nodes)

    # Matriz de restrições
    A = np.zeros((len(G.edges), n))
    for i, (u, v) in enumerate(G.edges):
        A[i, u] = 1
        A[i, v] = 1

    # Resolver a PL (minimizar custo sujeita a Ax >= 1)
    result = linprog(custos, A_ub=-A, b_ub=-np.ones(len(G.edges)), bounds=(0, 1))

    # Solução
    x = result.x

    # Selecionar vértices onde xv >= 1/2
    C = [v for v in G.nodes if x[v] >= 0.5]

    return C


def minCV_hochbaum_dual(G, c):
    # Número de vértices e arestas
    n = len(G.nodes)
    m = len(G.edges)

    # Matriz de restrições A (arestas para vértices)
    A = np.zeros((m, n))
    for i, (u, v) in enumerate(G.edges):
        A[i, u] = 1
        A[i, v] = 1

    # Resolver o problema dual com linprog
    result = linprog(-np.ones(m), A_ub=A.T, b_ub=c, bounds=(0, None), method='highs')

    if result.success:
        # Solução dual
        y = result.x

        # Selecionar vértices onde a igualdade na restrição dual se mantém
        C = []
        for v in G.nodes:
            total_dual_cover = sum(y[i] for i, (u, w) in enumerate(G.edges) if u == v or w == v)
            if total_dual_cover == c[v]:
                C.append(v)

        return C


def minCV_primal_dual(G, custos):
    # Inicializar o vetor dual y com valores zero para cada aresta
    y = {e: 0 for e in G.edges}

    # Função auxiliar para verificar se todas as arestas estão cobertas
    def all_edges_covered(G, custos):
        for v in G.nodes():
            if G.degree(v) == 0:  # Ignorar vértices isolados
                continue
            soma_y = 0
            for e in G.edges(v):  # Soma todas as arestas incidentes no vértice v
                soma_y += y[tuple(sorted(e))]  # Ordena a aresta para garantir consistência
            if soma_y < custos[v]:  # Se o custo de v não for atingido
                return False
        return True

    # Loop até que todas as arestas estejam cobertas
    while not all_edges_covered(G, custos):
        for e in G.edges():
            e = tuple(sorted(e))  # Ordenar a aresta
            a, b = e
            # Inicializar min_c como infinito para encontrar o incremento mínimo necessário
            min_c = float('inf')

            # Para cada vértice da aresta (a, b), calcular o quanto falta para cobrir o custo
            for v in [a, b]:
                if G.degree(v) == 0:  # Ignorar vértices isolados
                    continue
                soma_y = sum(y[tuple(sorted(edge))] for edge in G.edges(v))  # Soma das variáveis dual y
                if custos[v] - soma_y > 0:  # Se o custo de v não foi atingido
                    min_c = min(min_c, custos[v] - soma_y)

            # Incrementa y[e] apenas se min_c for positivo
            if min_c > 0 and min_c != float('inf'):
                y[e] += min_c  # Atualiza y para a aresta e

    # Selecionar os vértices onde a soma das arestas incidentes cobre o custo
    C = []
    for v in G.nodes():
        if G.degree(v) == 0:  # Ignorar vértices isolados
            continue
        soma_y = sum(y[tuple(sorted(e))] for e in G.edges(v))  # Soma das variáveis dual y
        if soma_y == custos[v]:  # Se o custo foi coberto
            C.append(v)

    return C


# Função para gerar um grafo aleatório grande com custos
def gerar_grafo_grande(n_vertices, n_arestas, custo_maximo):
    # Gerar um grafo aleatório com n_vertices e n_arestas
    G = nx.gnm_random_graph(n_vertices, n_arestas)

    # Gerar uma lista de custos aleatórios para cada vértice (entre 1 e custo_maximo)
    custos = [random.randint(1, custo_maximo) for _ in range(n_vertices)]

    return G, custos


# Gera um grafo aleatório
n_vertices = 15
n_arestas = random.randint(int(n_vertices/2), int(n_vertices*3))
custo_maximo = 10

G, custos = gerar_grafo_grande(n_vertices, n_arestas, custo_maximo)

# Exibindo informações básicas do grafo gerado
print(f"Número de vértices: {G.number_of_nodes()}")
print(f"Número de arestas: {G.number_of_edges()}")
print(f"Custos dos vértices: {custos}")

def draw_graph():
    nx.draw(G, with_labels=True)
    plt.show()

vertices = vertex_cover_exato(G, custos)
print(f"Solução Exata: {vertices} - {len(vertices)}")

C = minCV_hochbaum(G, custos)
print(f"Vertex Cover (Primal) 2-aproximativo: {C} - {len(C)}")

C2 = minCV_hochbaum_dual(G, custos)
print(f"Vertex Cover (Dual) 2-aproximativo: {C2} - {len(C2)}")

C3 = minCV_primal_dual(G, custos)
print(f"Vertex Cover (Primal-Dual) 2-aproximativo: {C3} - {len(C3)}")

draw_graph()