
from graph import Graph, graph_from_file, estimate_time, estimate_graph, UnionFind
from collections import deque

"""
data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)"""



from graph import Graph # on importe la classe graphe du fichier graph.py
#g = Graph([1,2]) # creation d!un objet de type Graph
#g.add_edge(1, 2, 4, 21)
#g = graph_from_file("input/network.1.in")
#print(g) # affichage du graphe
#print(g.get_path_with_power(2,4,4))
#print(g.min_power(20, 11))
#print(g.bfs(1,10,15))
#print(g.connected_components_set())
#print(g.min_power(3, 1))
print(estimate_time(2))
#estimate_graph("input/network.2.in")



