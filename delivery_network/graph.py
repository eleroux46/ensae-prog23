import random
import time
import math
import random 
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
        #self.list_parent=[]
        #self.list_rank=[]
        self.components_list=[]
        self.rank_dict= dict([(n, []) for n in nodes])
        self.parent_dict=dict([(n, []) for n in nodes])
        self.mst= dict([(n, []) for n in nodes])
        self.list_of_index = []
    

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
        visited=set()
        queue=deque()
        queue.append((depart, [depart]))
        while queue: 
            node, path=queue.popleft()
            neighbour_and_pwr = dict()
            for neighbour in self.graph[node]: 
                if neighbour[0] not in visited:
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
                    visited.add(adjacent_node)
       


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
        #mst.list_parent=uf.parent
        #mst.list_rank=uf.rank 
        #mst.components_list, mst.rank_dict, mst.parent_dict, _= self.build_characteristics()

        return mst
    



    def build_characteristics(self):
        """
        The function build_characteristics is applied to a graph and defines :
            -self.components_list: a list of the sets of the connected components of the mst graph
            -self.rank_dict: a dictionnary of the ranks of the nodes (format: {node:rank})
            -self.parent_dict: a dictionnary of the parents of the nodes (format: {node:parent})

        Its complexity is the one of the dfs2 : O(V+E') (V is the number of nodes and E' is the number of edges in the mst)
        and it is added to the complexity of the kruskal function which is used to form the mst.
        """
        #initialisations
        components_list=[]
        marked_sommet = {sommet:False for sommet in self.nodes}
        mst=self.kruskal()
        rank_dict={node:0 for node in self.nodes}
        parent_dict={node:0 for node in self.nodes}
        i=[0]

        def dfs2(sommet):
            #implementation of the dfs algorithm:
            component = [sommet]
            i[0]+=1 #i is used to measure the rank of the nodes

            for neighbour in mst.graph[sommet]:
                neighbour= neighbour[0]

                if not marked_sommet[neighbour] : #if the neighbour has not already been visited
                    marked_sommet[neighbour] = True
                    rank_dict[neighbour]= i[0]
                    parent_dict[neighbour]=sommet 
                    component += dfs2(neighbour)
            i[0]-=1
            
            return component
        
        
        for sommet in self.nodes:
            
            if not marked_sommet[sommet]:
                component=dfs2(sommet)
                component=set(component) #the list becomes a set in order to reduce the time complexity of the min_power_kruskal 
                components_list.append(component)
                            
        #the elements are assigned the the elements of the class graph                   
        self.components_list= components_list
        self.rank_dict = rank_dict
        self.parent_dict = parent_dict

        #return components_list, rank_dict, parent_dict


    def min_power_kruskal(self, src, dest):
        """
        the function takes a node of origin and a node of destination as parameters.
        it returns the p_min (which is the power minimal necessary to go from src to dest).
        
        Its complexity is : 
        """

        rank_src= self.rank_dict[src]
        rank_dest= self.rank_dict[dest]
        list_parents_dest=[dest]
        list_parents_src=[src]
        p_min=0


        for i in range(0, len(self.components_list)):
            if src and dest in self.components_list[i]: #checking to see if src and dest are connected
                
                #first we equalize the ranks 
                if rank_src < rank_dest:
                    while rank_src < rank_dest:
                        #the list of parents of dest is updated until rank_dest is equal to rank_src 
                        list_parents_dest.append(self.parent_dict[list_parents_dest[-1]])
                        rank_dest -=1

                elif rank_src > rank_dest:
                    while rank_src > rank_dest:
                        
                        list_parents_src.append(self.parent_dict[list_parents_src[-1]])           
                        rank_src -=1

                while list_parents_dest[-1] != list_parents_src[-1]:
                #loop until we find the lowest common ancestor                     

                    list_parents_dest.append(self.parent_dict[list_parents_dest[-1]])
                    list_parents_src.append(self.parent_dict[list_parents_src[-1]])
                

            else:
                return None 
   
        #building of the complete path between src and dest 
        list_src_rev=list_parents_src[:-1] 
        list_src_rev=list_src_rev[::-1] 
        path = list_parents_dest+list_src_rev

        #searching the min_power necessary to travel this path
        for index in range(len(path)-1):
            src, dest = path[index], path[index+1]
            dest_index= self.list_of_index[src-1].index(dest) 
            power= self.mst.graph[src][dest_index][1] #this is the power corresponding to the edge between the two nodes 
            if power > p_min:
                p_min=power #updating the p_min 

        return  p_min 
      
    





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
            dist = float(parameters[3])
        power_min = parameters[2].strip("\n")
        g.add_edge(int(parameters[0]), int(parameters[1]), int(power_min), dist)
    fil.close()
    g.build_characteristics()
    g.mst=g.kruskal()
    g.list_of_index = [list(zip(*g.mst.graph[node]))[0] if g.mst.graph[node]!=[] else () for node in g.nodes]
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
    nb_trajets =0
    for line in h:
        nb_trajets+=1
    h.close()
    h=open(f"input/routes.{argument}.in", "r")
    lignes=[]
    for i in range(nb_routes): 
        lignes.append(random.randint(1,nb_trajets-1)) #we randomly choose 10 lines to find the corresponding routes
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
    #estimation_time = mean_time_per_routes * len(g.nodes)
    estimation_time = mean_time_per_routes*nb_trajets
    print(f"Temps estimé : {estimation_time} secondes")
    return estimation_time


def estimate_time_kruskal(argument):
    """
    the function estimate_time takes the number of a network as argument
    and returns the mean time necessary to apply the min_power_kruskal function in this network.
    """
    g = graph_from_file(f"input/network.{argument}.in")
    h= open(f"input/routes.{argument}.in", "r") #we open the file "routes" corresponding 
    # initialisation of the time 
    total_time = 0
    # we want to test on 10 random routes
    nb_routes = 10
    nb_trajets =0
    for line in h:
        nb_trajets+=1
    h.close()
    h=open(f"input/routes.{argument}.in", "r")
    lignes=[]
    for i in range(nb_routes): 
        lignes.append(random.randint(1,nb_trajets-1)) #we randomly choose 10 lines to find the corresponding routes
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
            g.min_power_kruskal(src, dest) 
        except RecursionError:
            print("the function encountered a Recursion Error")
        end = time.perf_counter()
        total_time += end - start
    #calcul of the mean time per route:
    mean_time_per_routes = total_time / nb_routes

    # estimating the time necessary to calculate the min power on all of the routes
    #estimation_time = mean_time_per_routes * len(g.nodes)
    estimation_time = mean_time_per_routes*nb_trajets
    print(f"Temps estimé : {estimation_time} secondes")
    return estimation_time



def compare(argument):
    """
    this function takes the number of a network as argument 
    and returns the difference of the estimate time necessary to find the minimal power on this network with the function min_power and the function min_power_kruskal
    """
    time1= estimate_time(argument) 
    time2= estimate_time_kruskal(argument)

    print(f"La différence de temps estimée est de : {time1-time2} secondes")
    """
    test with argument = 2: 
    Temps estimé : 862529.8500002827 secondes
    Temps estimé : 6.288003642112017 secondes
    La différence de temps estimée est de : 862523.5619966406 secondes
    """



def stock_results(argument):
    """
    For each routes.x.in file, write a routes.x.out file that contains T lines
    with on each line :
        - a number corresponding to the minimum power to cover the path
        - a second number corresponding to the utility of the path 
    """
    file = open(f'input/routes.{argument}.in', 'r')
    g = graph_from_file(f'input/network.{argument}.in')
    output = open(f'output/routes.{argument}.out','w')
    output.write(file.readline())
    for line in file:
        list_line = line.split(' ')
        src = int(list_line[0])
        dest = int(list_line[1])
        utilite = list_line[2]
        p_min=g.min_power_kruskal(src,dest)
        if p_min==None:
            min_power = "None"
        else:
            min_power = p_min 
        output.write(str(min_power) + " " + str(utilite))
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




"""passage à l'algorithme d'optimisation avec les camions"""


  
class Catalogue:
    def __init__(self, trucks=[]):
        self.trucks = trucks
        self.catalogue = dict([(n, []) for n in trucks])
        self.nb_trucks = len(trucks)

    def __str__(self):
        if not self.catalogue:
            output = "The catalogue is empty"            
        else:
            output = f"The catalogue has {self.nb_trucks} trucks \n"
            for source, destination in self.catalogue.items():
                output += f"{source}-->{destination}\n"
        return output   
    
    def add_characteristics(self, truck, power, cost):
        if truck not in self.trucks:
            self.trucks[truck] = []
            self.nb_trucks += 1
            self.trucks.append(truck)
        self.catalogue[truck].append((power, cost))
    
    
def catalogue_from_file(filename):
    f = open(f"output/trucks.{filename}.in")
    content = f.readlines()
    nb_trucks = int(content[0])
    g = Catalogue([truck for truck in range(1, int(nb_trucks)+1)])
    g.nb_trucks = nb_trucks
    g.trucks = range(1, g.nb_trucks + 1)
    for line in range(1, int(nb_trucks)+1):
        parameters = (content[line]).split(" ")
        truck_power = parameters[0]
        truck_cost = parameters[1].strip("\n")
        g.add_characteristics(line, truck_power, truck_cost)
    return g


def journey_and_cost_algorithm(num_graph, num_catalogue):
    """
    Cette fonction construit un dictionnaire avec en clé : trajet (1,...nb_trajets) et des valeurs
    (camion choisit, son cout, utilité du trajet)"""
    with open(f'output/routes.{num_graph}.out', 'r') as fileout:
        nb_routes = int(fileout.readline())
        content_out = fileout.readlines()
    catal = catalogue_from_file(num_catalogue)
    journey_parameters = dict()
    #nb_routes = 0
    for index, line in enumerate(content_out):
        #nb_routes +=1
        parameters = line.split()
        utility = int(parameters[-1].strip("\n"))
        trajet_power = float(parameters[0])
       # print(trajet_power)
        if trajet_power == "None":
            continue
        opti_cost = float('inf')
        optimal_truck = None
        left = 0
        right = len(catal.trucks)-1
        while left <= right:
            mid = (left + right) // 2
            truck = catal.trucks[mid]
            for power, cost in catal.catalogue[truck]:
                if trajet_power <= float(power) and float(cost) < opti_cost:
                    opti_cost = int(cost)
                    optimal_truck = truck
                    break
            if float(power) >= trajet_power:
                right = mid - 1
            else:
                left = mid + 1
        journey_parameters[index+1] = [1, optimal_truck, opti_cost, utility]

    return journey_parameters, nb_routes, len(catal.trucks)


def glouton_algorithm(num_graph, num_catalogue):
    journey_parameters, nb_routes, nb_trucks = journey_and_cost_algorithm(num_graph, num_catalogue)
    #Trajet_and_utility = dict()
    for key, values in journey_parameters.items():
        #Trajet_and_utility[key]= (float(values[-1])/float(values[1]))
        journey_parameters[key].append(float(values[-1])/float(values[1]))
    # On sort les efficacité par ordre décroissant
    dict_sorted = dict(sorted(journey_parameters.items(),
                       key=lambda x: x[-1], reverse=True))
    # On initialise
    Cost_cumul = 0
    # On initialise la collection des camions à acheter
    Trucks_to_buy = dict()
    routes_and_trucks = dict()
    total_utility = 0
    print(dict_sorted)
    # on itère sur nb_routes et pas sur len(dict_sorted) car dans le dict sorted les routes inexistantes ont été suppr
    for i in dict_sorted:
        try:
            if float(dict_sorted[i][1]) + Cost_cumul <= 25*(10**9):
                Cost_cumul += int(dict_sorted[i][2])
                # on construit une liste avec : tel trajet : si il est cover : et si oui par qui
                routes_and_trucks[i] = [1, dict_sorted[i][1], dict_sorted[i][2], dict_sorted[i][3]]
                total_utility += int(dict_sorted[i][3])
                if dict_sorted[i][0] not in Trucks_to_buy.keys():
                    Trucks_to_buy[dict_sorted[i][1]] = 1
                else:
                    Trucks_to_buy[dict_sorted[i][1]] += 1
            else:
                routes_and_trucks[i] = [0, dict_sorted[i][1], dict_sorted[i][2], dict_sorted[i][3]]
        except KeyError:
            pass  # le trajet i n'existe pas, n'est pas possible
            #solution.routes[index+1] = [opti_truck, opti_cost, utility]
    return Trucks_to_buy, routes_and_trucks,total_utility ,Cost_cumul, nb_trucks 


class Solution:
    def __init__(self, trucks, routes):
        self.trucks = trucks
        self.routes = routes
        self.cost = 0
        self.utility = 0

    def __str__(self):
        return f"Trucks: {self.trucks}\nRoutes: {self.routes}\nCost: {self.cost} & Utility: {self.utility}"


def Simulated_annealing_random(num_graph, num_catalogue, nb_iter=1000, T=1000, alpha=0.95):

    #Initialisation de type random:
    with open(f'routes.{num_graph}.out', 'r') as fileout:
        nb_routes = int(fileout.readline())
    result = journey_and_cost_algorithm(num_graph,num_catalogue)
    nb_trucks = result[-1]
    routes = result[0]
    routes_sorted = dict(sorted(routes.items(),key=lambda x: x[-1], reverse=True))
    solution = Solution(trucks=list(range(1, nb_trucks +1)), routes= routes_sorted)
    utility = 0
    cost = 0
    solution.utility = utility
    solution.cost = cost
    
    print(solution)
    best_solution = solution
    for i in range(nb_iter):
        print(i)
        # Générer une solution voisine
        new_solution = Solution(trucks=best_solution.trucks.copy(), routes=best_solution.routes.copy())
        for j in range(1,10) : 
            change1, change2 = random.randint(1, len(new_solution.routes)), random.randint(1, len(new_solution.routes))
            if new_solution.routes[change1][0]== 0 :
                new_solution.routes[change1][0] = 1
            if new_solution.routes[change2][0]== 1 :
                new_solution.routes[change2][0] = 0
        #print("voici change1 :", change1, new_solution.routes[change1][0]," voici change2 :" ,change2, new_solution.routes[change2][0])
        #print(new_solution)
        # Calculer le coût et la profitabilité de la nouvelle solution
        new_solution.cost = 0
        new_solution.utility = 0
        for k in new_solution.routes:
            #print(f"boucle numéro {k} on a un {new_solution.routes[k][0]}")
            #print(f"{k} : {new_solution.routes[k][0]} ----")
            if new_solution.routes[k][0] == 1 and float(new_solution.routes[k][2]) + new_solution.cost <= 25*(10**9): 
                new_solution.cost += new_solution.routes[k][2]
               #     print("nouveau new_solution.cost", new_solution.cost)
                new_solution.utility += new_solution.routes[k][-1]
                
        # Accepter ou rejeter la nouvelle solution selon la probabilité de Metropolis
        if new_solution.utility > solution.utility or math.exp((new_solution.utility - solution.utility) / T) > random.uniform(0, 1):
            solution = new_solution

        # Mettre à jour la meilleure solution trouvée
        print("ma solution : ",new_solution.utility)
        
        if solution.utility > best_solution.utility:
            best_solution = new_solution
        print("meilleure solution utility", best_solution.utility)
        # Refroidir la température
        T *= alpha
    print()
    print("Solution trouvée :", best_solution)
    return best_solution


def Simulated_annealing_glouton(num_graph, num_catalogue, nb_iter=1000, T=1000, alpha=0.95):
    with open(f'output/routes.{num_graph}.out', 'r') as fileout:
        nb_routes = int(fileout.readline())
        content_out = fileout.readlines()
    # Initialisation de type 1 :
    catal = catalogue_from_file(num_catalogue)
    result = glouton_algorithm(num_graph,num_catalogue)
    nb_trucks = result[-1]
    routes = result[1]
    print("ON EST LA")
    routes_sorted = dict(sorted(routes.items(),key=lambda x: x[-1], reverse=True))
    solution = Solution(trucks=list(range(1, nb_trucks +1)), routes = routes_sorted)
    utility = result[2]
    cost = result[3]
    solution.utility = utility
    solution.cost = cost
    
    #print(solution)
    best_solution = solution
    for i in range(nb_iter):
        print(i)
        # Générer une solution voisine
        new_solution = Solution(trucks=best_solution.trucks.copy(), routes=best_solution.routes.copy())
        for j in range(1,10) : 
            change1, change2 = random.randint(1, len(new_solution.routes)), random.randint(1, len(new_solution.routes))
            if new_solution.routes[change1][0]== 0 :
                new_solution.routes[change1][0] = 1
            if new_solution.routes[change2][0]== 1 :
                new_solution.routes[change2][0] = 0
        #print("voici change1 :", change1," voici change2 :" ,change2)
        #print(new_solution)
        # Calculer le coût et la profitabilité de la nouvelle solution
        new_solution.cost = 0
        new_solution.utility = 0
        for k in new_solution.routes:
            #print(f"boucle numéro {k} on a un {new_solution.routes[k][0]}")
            #print(f"{k} : {new_solution.routes[k][0]} ----")
            if new_solution.routes[k][0] == 1 and float(new_solution.routes[k][2]) + new_solution.cost <= 55*(10**5): 
                new_solution.cost += new_solution.routes[k][2]
               #     print("nouveau new_solution.cost", new_solution.cost)
                new_solution.utility += new_solution.routes[k][-1]
                
        # Accepter ou rejeter la nouvelle solution selon la probabilité de Metropolis
        if new_solution.utility > solution.utility or math.exp((new_solution.utility - solution.utility) / T) > random.uniform(0, 1):
            solution = new_solution

        # Mettre à jour la meilleure solution trouvée
        print("ma solution : ",new_solution.utility)
        
        if solution.utility > best_solution.utility:
            best_solution = new_solution
        print("meilleure solution utility", best_solution.utility)
        # Refroidir la température
        T *= alpha
    print()
    print("Solution trouvée :", best_solution)
    return best_solution