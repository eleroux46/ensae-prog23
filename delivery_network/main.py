from graph import Graph, graph_from_file
from collections import deque

"""
data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)"""



from graph import Graph # on importe la classe graphe du fichier graph.py
#g = Graph([1,2]) # creation d!un objet de type Graph
#g.add_edge(1, 2, 4, 21)
g = graph_from_file("input/network.03.in")
print(g) # affichage du graphe
#print(g.get_path_with_power(1,2,5))
print(g.min_power_kruskal(1,4))

