# problems with closure:
# n: elements with the same name are not deemed as the same elements, i.e. in [('C', 'A'), ('C', 'A')] it will detect 4 elements

from src.finetuning.graph_structure import Graph

def lower(v, graph) -> int:
    unique = graph.unique_elements_set()
    if v in unique:
        unique.remove(v)
    return sum(1 for u in unique if graph.find_path(v, u))

def upper(v, graph, n) -> int:
    unique = graph.unique_elements_set()
    if v in unique:
        unique.remove(v)
    return n - sum(1 for u in unique if graph.find_path(u, v))


def strength(v, graph, n) -> float:
    res = (upper(v, graph, n) + lower(v, graph)) / (2 * n)
    return res

def edge_score(edge, graph) -> int:
    u, v = edge
    old_len = graph.length()
    #print("old_len", old_len)
    #print("old_graph", graph.view_as_tuples())
    new_graph = Graph(set(graph.view_as_tuples()))
    new_graph.add_edge(u, v)
    #print("new_graph", new_graph.view_as_tuples())
    transitive_closure = new_graph.transitive_closure()
    #print("transitive_closure", transitive_closure.view_as_tuples())
    new_len = transitive_closure.length()
    #print("new_len", new_len)
    #print((new_len - old_len) / max(old_len, 1))
    #return (new_len - old_len)
    return (new_len - old_len) / max(old_len, 1)

def stronger_likelihood(edge, graph, n) -> float:
    u, v = edge
    return 0.5 + (strength(u, graph, n) - strength(v, graph, n)) / 2


def pair_score(pair, graph, n):
    u, v = pair
    p = stronger_likelihood((u, v), graph, n)
    #print(p)
    #print(edge_score((u, v), graph))
    #print(1 - p)
    #print(edge_score((v, u), graph))
    return p * edge_score((u, v), graph) + (1 - p) * edge_score((v, u), graph)


def find_appropriate_candidates(node_with_score: tuple, elements_with_scores):
    v, score = node_with_score
    if score < elements_with_scores[0][1]:
        return None, elements_with_scores[0][0]
    elif score > elements_with_scores[len(elements_with_scores)-1][1]:
        return elements_with_scores[len(elements_with_scores)-1][0], None
    else:
        counter = 0
        while counter < len(elements_with_scores-1):
            if elements_with_scores[counter][1] <= score <= elements_with_scores[counter + 1][1]:
                return elements_with_scores[counter][0], elements_with_scores[counter + 1][0]
            else:
                counter += 1


def calculate_candidate_score(n, pair, graph, elements_with_scores):
    u, v = pair
    if graph.connected(u, v):
        return 0
    unique_elements = [element[0] for element in elements_with_scores]
    if u or v in unique_elements:
        return pair_score(pair, graph, n)
    else:
        score_u = strength(u, graph, n)
        score_v = strength(v, graph, n)
        lower_node_u, upper_node_u = find_appropriate_candidates((u, score_u), elements_with_scores)
        lower_node_v, upper_node_v = find_appropriate_candidates((v, score_v), elements_with_scores)
        if not lower_node_u:
            score_u = pair_score((u, upper_node_u), graph, n)
        elif not upper_node_u:
            score_u = pair_score((u, lower_node_u), graph, n)
        else:
            score_u = (pair_score((u, upper_node_u), graph, n) + pair_score((u, lower_node_u), graph, n)) / 2
        if not lower_node_v:
            score_v = pair_score((v, upper_node_v), graph, n)
        elif not upper_node_v:
            score_v = pair_score((v, lower_node_v), graph, n)
        else:
            score_v = (pair_score((v, upper_node_v), graph, n) + pair_score((v, lower_node_v), graph, n)) / 2
        score = (score_u + score_v) / 2

        return score

def calculate_scores(graph, n):
    elements = graph.unique_elements_set()
    # print(elements)
    elements_with_scores = []
    for element in elements:
        score = strength(element, graph, n)
        # print(score)
        elements_with_scores.append((element, score))
    return sorted(elements_with_scores, key=lambda x: x[1])


# elements_with_scores = calculate_scores(E_graph, n)


#print("SCORE", pair_score(('C', 'E'), E_graph))
#print("SCORE", pair_score(('B', 'E'), E_graph))
#print("SCORE", pair_score(('C', 'C'), E_graph))


"""
######################################
# https://stackoverflow.com/questions/8673482/transitive-closure-python-tuples
def transitive_closure(a):
    closure = set(a)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
        closure_until_now = closure | new_relations
        if len(closure_until_now) == len(closure):
            break
        closure = closure_until_now
    return closure


# dummy data
V = set('ABCDEF')
n = len(V)
# E = [
#     ('B', 'A'),
#     ('C', 'B'),
#     ('C', 'E'),
#     ('D', 'E'),
#     ('E', 'F'),
# ]
# E = [
#     ('A', 'B'),
#     ('B', 'C'),
#     ('E', 'C'),
#     ('E', 'D'),
#     ('F', 'E'),
# ]

E = [
    ('B', 'C'),
    ('C', 'A'),
    ('E', 'D')
]

E = transitive_closure(E)

# (u, v) in E means "v > u"
# (u, v) in E means "u > v"
print(E)
m = len(E)


def upper(v, closure) -> int:
    return n - sum(1 for u in V if (u, v) in closure)

print(upper('B', E))

def lower(v, closure) -> int:
    return sum(1 for u in V if (v, u) in closure)


def strength(v, closure) -> float:
    res = (upper(v, closure) + lower(v, closure)) / (2 * n)
    print(v, res)
    return res



def edge_score(edge, closure) -> int:
    old_len = len(closure)
    new_len = len(transitive_closure(closure.union({edge})))
    return (new_len - old_len) // max(old_len, 1)


def stronger_likelihood(edge, closure) -> float:
    u, v = edge
    return 0.5 + (strength(v, closure) - strength(u, closure)) / 2


def pair_score(pair, closure):
    u, v = pair
    p = stronger_likelihood((u, v), closure)
    return p * edge_score((u, v), closure) + (1 - p) * edge_score((v, u), closure)


labels = set()
# pool = set(tuple(sorted(edge)) for edge in E)
pool = set(tuple(edge) for edge in E)
print(pool)
#print(pool.difference_update(tuple(sorted(l)) for l in labels))
labels = transitive_closure(labels)
print(labels)

scores = sorted(((pair_score(pair, labels), pair) for pair in pool), reverse=True)
print('Scores:   ', scores)
max_score, sel_pair = scores[0]
print(max_score)
print(sel_pair)

# while len(labels) < len(E):
    # print('=' * 40)
    # labels = transitive_closure(labels)
    # print('Labels:   ', labels)
    # pool.difference_update(tuple(sorted(l)) for l in labels)
    # print('Pool:     ', pool)
    # scores = sorted(((pair_score(pair, labels), pair) for pair in pool), reverse=True)
    # print('Scores:   ', scores)
    # max_score, sel_pair = scores[0]
    # print('Selection:', sel_pair)
    # if sel_pair in E:
    #     labels.add(sel_pair)
    # else:
    #     sel_pair = sel_pair[::-1]
    #     assert sel_pair in E
    #     labels.add(sel_pair)
    # print('New Label:', sel_pair)
"""