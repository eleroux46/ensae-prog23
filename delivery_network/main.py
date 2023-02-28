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
g = graph_from_file("input/network.02.in")
print(g) # affichage du graphe
#print(g.get_path_with_power(1,2,5))
#print(g.bfs(1,2,5))

path=[]
queue=deque()
queue.append((1, [1]))
while queue:
    node, path=queue.popleft()
    if len(g.graph[1])>1:
        adjacent_nodes = []
        power_min= []
        for j in range(len(g.graph[1])):
            adjacent_nodes.append(g.graph[1][j][0])
            power_min.append(g.graph[1][j][1])
            for i in range(len(adjacent_nodes)):
                if adjacent_nodes[i]==2 and power_min[i]<=5:
                    path=path+[adjacent_nodes[i]]
                elif power_min[i]<=5:
                    queue.append((adjacent_nodes[i], path+[adjacent_nodes[i]]))
                else:
                    path=path
print(path)
