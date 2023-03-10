import random
import time
from typing import List, Tuple, Dict, Optional
import heapq
from collections import deque
class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
            
        
        The __init__() function initializes the graph with a set of nodes and no edges. It takes an optional list nodes as a parameter, which contains the nodes of the graph. If nodes is not provided, it is initialized to an empty list. The graph attribute is initialized to an empty dictionary, where the keys are the nodes and the values are the adjacency lists of each node. The nb_nodes attribute is initialized to the length of nodes, and the nb_edges attribute is initialized to 0.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)
        
        The __str__() function returns a string representation of the graph, showing the adjacency list of each node. If the graph is empty, it returns a string saying so. Otherwise, it returns a string showing the number of nodes and edges, followed by each node and its adjacency list.
        
        """
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 
        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
    

    def connected_components(self):
        """
        The connected_components() function finds the connected components of the graph using depth-first search. It returns a list of lists, where each inner list contains the nodes of a connected component.
        The time complexity of this function is O(V+E), where V is the number of nodes and E is the number of edges. This is because the function visits each node and each edge exactly once using depth-first search.
        """
        components_list = []
        marked_sommet = {sommet:False for sommet in self.nodes}
        def dfs(sommet):
            component = [sommet]
            for neighbour in self.graph[sommet]:
                neighbour = neighbour[0]
                if not marked_sommet[neighbour]:
                    marked_sommet[neighbour] = True
                    component += dfs(neighbour)
            return component

        for sommet in self.nodes:
            if not marked_sommet[sommet]:
                components_list.append(dfs(sommet))
                
        return components_list



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    


        
    def bfs(self, depart, fin, power):
        #erreur = 0 
        path=[]
        i = 0 
        queue=deque()
        queue.append((depart, [depart]))
        while queue: 
            i+=1
            node, path=queue.popleft()
            neighbour_and_pwr = dict()
            for neighbour in self.graph[node]: 
                if neighbour[0] not in path:
                    neighbour_and_pwr[neighbour[0]] = neighbour[1]
            to_remove = []  # initialisation de la liste des noeuds à supprimer
            for element, values in neighbour_and_pwr.items():
                if values > power:  # si la puissance du lien est supérieure à power
                    to_remove.append(element)  # on stocke l'élément dans la liste à supprimer
            for element in to_remove:
                del neighbour_and_pwr[element]  # on supprime les éléments stockés dans la liste à supprimer
            for adjacent_node in neighbour_and_pwr.keys():
                #erreur += 1
                if adjacent_node == fin:
                    return path + [adjacent_node]
                else:
                    queue.append((adjacent_node, path + [adjacent_node]))
       


    def get_path_with_power(self, src, dest, power):
        """Should return path.
        path is a list of nodes representing the path from src to dest that requires at least power power.
        The result should be the path with the smallest number of nodes and that requires at least power power.
        If there is no such path, the function should return None.
        
        The time complexity of the get_path_with_power() function is O(Elog(V)), where V is the number of nodes and E is the number of edges. This is because the function uses Dijkstra's algorithm, which has a time complexity of O(Elog(V)).
        """
        chemin = []
        connected=self.connected_components()
        for i in range(0, len(connected)):
            if src and dest in connected[i]:
            #marked_sommet = {sommet:False for sommet in connected[i]} #O(n): complexite a peu pres le nb de noeuds dans le graphe 
                chemin= self.bfs(src, dest, power)                
            else : 
                None      
            #if dest not in chemin:
               # chemin = None           
        return chemin
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        path is a list of nodes representing the path from src to dest.
        min_power is the minimum power on the path from src to dest.
        The result should be the path and the minimum power on the path from src to dest.
        If there is no path from src to dest, the function should return None, None.
        
        The time complexity of the min_power() function depends on the implementation. If you use Dijkstra's algorithm to find the shortest path, the time complexity is also O(E*log(V)). However, you can also use a brute-force approach that enumerates all paths from src to dest, and compute their minimum power. In this case, the time complexity would be O(2^V * E), which is exponential in the number of nodes.
        """
        #Should return path and min_power if there is a path from src to dest
        #Uses the function get_path_with_power but instead of taking the minimum power as a parameter and returning the path if a path is possible, it takes only the src and dest as parameters and returns, if possible, the path as well as the min_power needed. 

        path = self.get_path_with_power(src, dest, float('inf'))
        if path is None:
            return print("There is no possible path between these nodes")
        path = self.get_path_with_power(src, dest, 0) # Chemin avec puissance minimale
    
        # Si on peut atteindre la destination avec une puissance nulle, la réponse est 0
        if path is not None:
            return path, 0
    
        # Sinon, on fait une recherche binaire sur l'intervalle [0, max_power]
        low = 1
        high = max(edge[1] for node_edges in self.graph.values() for edge in node_edges)
        high = int(high)
        while low < high:
            mid = (low + high) // 2
            path = self.get_path_with_power(src, dest, mid)
            if path is not None:
                high = mid
            else:
                low = mid + 1
            
        # La puissance minimale est high
        return self.get_path_with_power(src, dest, high), high
    
    
    
    def get_path_with_power_and_distance(self, t: Tuple[str, str, int]) -> Optional[Tuple[List[str], int]]:
   
    #La complexité de cette fonction est la même que celle de la fonction get_path_with_power(), soit en O(m + n log n) dans le pire des cas, où m est le nombre d'arêtes dans le graphe et n est le nombre de noeuds. La seule différence est que nous effectuons également une sommation de la distance parcourue pour calculer la distance totale du chemin. Cela ajoute une complexité constante à chaque itération de la boucle principale, donc la complexité de la fonction reste en O(m + n log n).
        start, end, min_power = t
        queue = [(start, [start], 0, 0)]  # (node, path, power, distance)
        shortest_path = None
        
        while queue:
            node, path, power, distance = queue.pop(0)
            
            if node == end and power >= min_power:
                if shortest_path is None or distance < shortest_path[1]:
                    shortest_path = (path, distance)
                continue
            
            for neighbor, neigh_power, neigh_distance in self.graph[node]:
                if neighbor not in path and power >= neigh_power:
                    new_path = path + [neighbor]
                    new_power = power - neigh_power
                    new_distance = distance + neigh_distance
                    queue.append((neighbor, new_path, new_power, new_distance))
        
        return shortest_path
    
    
    def kruskal(self):
        start=time.perf_counter()
        uf=UnionFind(self.nb_nodes)
        #trier les aretes par ordre croissant:
        edges=[(power, src, dest) for src in self.nodes for dest, power, _ in self.graph[src]]
        edges.sort()
        #former l'arbre couvrant de puissance minimale:
        mst=Graph(self.nodes)
        for power, src, dest in edges:
            #finds the sets that contain src and dest
            src_set= uf.find(src)
            dest_set= uf.find(dest)
            if src_set != dest_set:
                mst.add_edge(src, dest, power)
                uf.union(src_set, dest_set)
        end=time.perf_counter()
        print(end-start)
        return mst
    
    def min_power_kruskal(self, src, dest):
        "apply kruskal fonction to a graph and return the path and the min power of the path between src and dest "
        mst=self.kruskal()
        return mst.min_power(src, dest)
        









def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    fil = open(filename,"r")
    content = fil.readlines()
    dist=1
    firstfil = content[0]
    firstfil2 = firstfil.split(" ")
    g = Graph([node for node in range(1,int(firstfil2[0])+1)])
    g.nb_nodes = int(firstfil2[0])
    g.nodes = range(1,g.nb_nodes + 1)
    for ligne in range(1, len(content)):
        parameters = (content[ligne]).split(" ")
        if len(parameters) == 4:
            dist = int(parameters[3])
        power_min = parameters[2].strip("\n")
        g.add_edge(int(parameters[0]), int(parameters[1]), int(power_min), dist)
    fil.close()
    return g


def estimate_time(filename):
    # créer une instance de la classe Graph à partir du fichier contenant les nœuds, chemins, et poids des chemins. 
    g = graph_from_file(filename)
    # mesurer le temps nécessaire pour calculer la puissance minimale et le chemin associé pour chaque trajet
    total_time = 0
    # On veut faire le test sur 10 trajets aléatoires
    nb_routes = 10
    for trajet in range(nb_routes):
        start = time.perf_counter()
        src, dest = random.sample(g.nodes, 1), random.sample(g.nodes, 1)
        print(g.min_power(src, dest))
        end = time.perf_counter()
        total_time += end - start
        print(src, dest)
    # calculer le temps moyen par trajet
    mean_time_per_routes = total_time / nb_routes

    # estimer le temps nécessaire pour calculer la puissance minimale (et le chemin associé) sur l'ensemble des trajets
    estimation_time = mean_time_per_routes * len(g.nodes)
    print(f"Temps estimé : {estimation_time} secondes")

def estimate_graph(filename):
    start = time.perf_counter()
    g= graph_from_file(filename)
    end = time.perf_counter()
    total_time = end - start
    print(total_time)


class UnionFind:
    def __init__(self,nb_nodes):
        self.parent = list(range(int(nb_nodes)+1))
        self.rank=[0]*(nb_nodes+1)

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node]=self.find(self.parent[node])
        return self.parent[node]

    def union(self, src, dest):
        src_root = self.find(src)
        dest_root = self.find(dest)
        if src_root==dest_root:
            return
        if self.rank[src_root]<self.rank[dest_root]:
            self.parent[src_root]=dest_root
        elif self.rank[src_root]>self.rank[dest_root]:
            self.parent[src_root]=dest_root
        else:
            self.parent[src_root]=dest_root
            self.rank[src_root]+=1