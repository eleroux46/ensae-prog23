import random
import time
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
        The connected_components function uses the depth-first search algorithm (parcours en profondeur). 
        It takes a graph as parameter and returns a list of connected components (a list of lists).
        The time complexity of this function is O(V+E) (V is the number of nodes and E is the number of edges). This is due to the use of dfs.
        """
        components_list = []
        marked_sommet = {sommet:False for sommet in self.nodes}
        #implementation of the dfs algorithm:
        def dfs(sommet):
            component = [sommet]
            for neighbour in self.graph[sommet]:
                neighbour = neighbour[0]
                if not marked_sommet[neighbour]:
                    marked_sommet[neighbour] = True
                    component += dfs(neighbour)
            return component
        
        #building of the connected_components list:
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
        """
        The function bfs (breadth-first search, ou parcours en largeur) takes a node of origin, of end and a power.
        It should return the path from depart to fin under a condition of power.
        Its complexity is at worse O(V+E) when each node and each edge is visited once.
        """
        #initialisation
        path=[]
        queue=deque()
        queue.append((depart, [depart]))
        while queue: 
            node, path=queue.popleft()
            neighbour_and_pwr = dict()
            for neighbour in self.graph[node]: 
                if neighbour[0] not in path:
                    neighbour_and_pwr[neighbour[0]] = neighbour[1]
            # initialisation of the list of nodes to be removed
            to_remove = [] 
            for element, values in neighbour_and_pwr.items():
                # if the power of the edge is higher than our condition of power
                if values > power:  
                    to_remove.append(element)  # then the node is stocked in our list of nodes to be removed
            # the nodes in our list "to_remove" are deleted from our dictionnary
            for element in to_remove:
                del neighbour_and_pwr[element]  
            #for each node remaining in our dictionnary of nodes to be visited
            for adjacent_node in neighbour_and_pwr.keys(): 
                if adjacent_node == fin: 
                    return path + [adjacent_node]
                else:
                    queue.append((adjacent_node, path + [adjacent_node])) #the node is added to the path 
       


    def get_path_with_power(self, src, dest, power):
        """
        this function returns a path, which is a list of nodes representing the path from src to dest under a condition of power.
        This function uses the function bfs in order to find a path.
        If there is no path under the condition of power, it returns None.
        
        The time complexity of the get_path_with_power() function is at worse O(V+E), the same time complexity as bfs.
        """
        chemin = []
        connected=self.connected_components() #the algorithm is only applied to the nodes connected with each other
        for i in range(0, len(connected)):
            if src and dest in connected[i]: #if the origin and the destination are connected
                chemin= self.bfs(src, dest, power) #the function bfs is applied in order to find the path between them           
            else : 
                None  #if they are not connected, there is no path between them              
        return chemin
    
    def min_power(self, src, dest):
        """
        this function takes a node of origin and a node of destination as parameters and returns the minimal power necessary for the path.
        If there is no such path, min_power returns None.
        This function uses the function get_path_with_power() which has a complexity O(V+E) and a binary search algorithm which has a complexity of O(ln(high)).
        So the total complexity of the function is O((V+E)*ln(high)).
        """
        path = self.get_path_with_power(src, dest, float('inf'))
        if path is None:
            return print("There is no possible path between these nodes")
        
        path = self.get_path_with_power(src, dest, 0)
        # if there is a path with the minimal power (0) then the function returns the path and 0
        if path is not None:
            return path, 0

        #else, we implement a binary search on the interval [0, max_power]
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
            
        # the minimal power needed is "high"
        return self.get_path_with_power(src, dest, high), high
    
    
    def kruskal(self):
        """
        The kruskal function takes as input a graph in Graph format and returns another element of this class.
        It transform a graph in a mst (minimum spanning tree) which is a graph whose sum of edge weights is minimal.
                                                                                             
        The complexity of kruskal is O(Eln(E) + Eln(V)) because :
            - First, the creation of n sets has a complexity of O(n)
            - Then, the sorting of the E edges has a complexity of O(Eln(E))
            - Finally, the "Union-find" of two sets has a complexity of O(ln(V))
        The complexity depends on the size of the graph and the number of edges
        """
        #first, a set is created for each node
        uf=UnionFind(self.nb_nodes)
        #then, the edges are sorted:
        edges=[(power, src, dest) for src in self.nodes for dest, power, _ in self.graph[src]]
        edges.sort()
        #finally, the minimum spanning tree is initialised (arbre couvrant de poids minimal) 
        mst=Graph(self.nodes)
        for power, src, dest in edges:
            #the mst is built using the functions union-find
            src_set= uf.find(src)
            dest_set= uf.find(dest)
            if src_set != dest_set: #if the nodes are not already connected, the edge is added to the mst
                mst.add_edge(src, dest, power)
                uf.union(src_set, dest_set)
        return mst
    



    def min_power_kruskal(self, src, dest):
        """
        the function takes a node of origin and a node of destination as parameters.
        it returns the min_power of the minimum spanning tree corresponding to the graph (using the kruskal function).
        The complexity of the min_power_kruskal function is O(Eln(E)+Eln(V) + (V+E)*ln(high)) (where high is the power_max) because it uses the functions krukal and min_power.
        But 
        """
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


def estimate_time(argument):
    """
    the function estimate_time takes the number of a network as argument
    and returns the mean time necessary to apply the min_power function in this network.
    """
    g = graph_from_file(f"input/network.{argument}.in")
    h= open(f"input/routes.{argument}.in", "r") #we open the file "routes" corresponding 
    # initialisation of the time 
    total_time = 0
    # we want to test on 10 random routes
    nb_routes = 10
    lignes=[]
    for i in range(nb_routes): 
        lignes.append(random.randint(1,nb_routes-1)) #we randomly choose 10 lines to find the corresponding routes
    lignes.append(0)
    #the 10 routes are sorted
    lignes.sort()
    for trajet in range(nb_routes):
        for i in range(lignes[trajet+1]-lignes[trajet]):
            h.readline() #the lines unused are read
        src, dest,_ = h.readline().split() #we retrieve the route we want to test 
        src=int(src)
        dest=int(dest)
        start = time.perf_counter()
        try:
            g.min_power(src, dest)
        except RecursionError:
            print("the function encountered a Recursion Error")
        end = time.perf_counter()
        total_time += end - start
    #calcul of the mean time per route:
    mean_time_per_routes = total_time / nb_routes

    # estimating the time necessary to calculate the min power on all of the routes
    estimation_time = mean_time_per_routes * len(g.nodes)
    print(f"Temps estimé : {estimation_time} secondes")
    return estimation_time

def compare(argument):
    g= graph_from_file(f"input/network.{argument}.in")
    time1= estimate_time(argument)

    # we use the same structure as the estimate_time function

    h= open(f"input/routes.{argument}.in", "r") #we open the file "routes" corresponding 
    # initialisation of the time 
    total_time = 0
    # we want to test on 10 random routes
    nb_routes = 10
    lignes=[]
    for i in range(nb_routes): 
        lignes.append(random.randint(1,nb_routes-1)) #we randomly choose 10 lines to find the corresponding routes
    lignes.append(0)
    #the 10 routes are sorted
    lignes.sort()
    for trajet in range(nb_routes):
        for i in range(lignes[trajet+1]-lignes[trajet]):
            h.readline() #the lines unused are read
        src, dest,_ = h.readline().split() #we retrieve the route we want to test 
        src=int(src)
        dest=int(dest)
        start = time.perf_counter()
        g.min_power_kruskal(src, dest)
        end = time.perf_counter()
        total_time += end - start
    #calcul of the mean time per route:
    mean_time_per_routes = total_time / nb_routes

    # estimating the time necessary to calculate the min power on all of the routes
    estimation_time = mean_time_per_routes * len(g.nodes)
    print(f"La différence de temps estimée est de : {time1-estimation_time} secondes")


def stock_results(argument):
    """
    For each routes.x.in file, write a routes.x.out file that contains T lines
with on each line a single number corresponding to the minimum power to cover the path
    """
    file = open(f'input/routes.{argument}.in', 'r')
    g = graph_from_file(f'input/network.{argument}.in')
    kruskal = g.kruskal()
    output = open(f'output/routes.{argument}.out','w')
    file.readline()
    for line in file:
        list_line = line.split(' ')
        src = int(list_line[0])
        dest = int(list_line[1])
        if kruskal.min_power_kruskal(src,dest)==None:
            min_power = "None"
        else:
            min_power = kruskal.min_power_kruskal(src,dest)[-1]
        output.write(str(min_power))
        output.write('\n')
    output.close()


        




class UnionFind:
    def __init__(self,nb_nodes):
        self.parent = list(range(int(nb_nodes)+1))
        self.rank=[0]*(nb_nodes+1)

    def find(self, node):
        """
        the function finds the parent of a node
        """
        if self.parent[node] != node:
            self.parent[node]=self.find(self.parent[node])
        return self.parent[node]

    def union(self, src, dest):
        """
        this function is used to merge the roots of two nodes
        """
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
