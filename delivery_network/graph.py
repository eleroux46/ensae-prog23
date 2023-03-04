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
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
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


        self.nb_edges+=1
        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))

    def dfs(self, sommet):
        component = [sommet]
        for neighbour in self.graph[sommet]:
            neighbour = neighbour[0]
            if not marked_sommet[neighbour]: #O(n+m): visite chq arrete et chq sommet une fois 
                marked_sommet[neighbour] = True
                component += dfs(neighbour)
        return component
    

    def dfs2(self, sommet, sommet2, power, visites):
        component = [sommet]
        for neighbour in self.graph[sommet]:
            neighbour1 = neighbour[0]
            power_neighbour= neighbour[1]
            visites[sommet]=True
            if not visites[neighbour1] and power_neighbour <= power and component[-1]!=sommet2 : #O(n+m): visite chq arrete et chq sommet une fois 
                visites[neighbour1] = True
                component += self.dfs2(neighbour1, sommet2, power, visites)
        return component
#bfs : parcours en largeur 
#garder le parent de chacun des noeuds 

#définition d'un parcours en largeur (bfs) :
    def bfs(self, depart, fin, power):
        path=[]
        erreur=0
        queue=deque()
        queue.append((depart, [depart]))
        while queue: #bfs de complexite O(n+m) 
            node, path=queue.popleft()
            adjacent_nodes = []
            path_pwr = []
            for neighbour in self.graph[node] :
                if neighbour[0] not in path:
                    adjacent_nodes.append(neighbour[0])
                    path_pwr.append(neighbour[1])
            for element in path_pwr: 
                if power < int(element): 
                    del adjacent_nodes[path_pwr.index(element)], path_pwr[path_pwr.index(element)]
            for adjacent_node in adjacent_nodes:
                if adjacent_node == fin:
                    return path + [adjacent_node]
                else:
                    queue.append((adjacent_node, path + [adjacent_node]))

        return path



    def get_path_with_power(self, src, dest, power):
        #src= source= noeud de depart 
        #dest= destination

        #on cherche a savoir si le chemin est possible
        connected=self.connected_components()                
        chemin = []
        for i in range(0, len(connected)):
            if src and dest in connected[i]:
                #marked_sommet = {sommet:False for sommet in connected[i]} #O(n): complexite a peu pres le nb de noeuds dans le graphe 
                chemin= self.bfs(src, dest, power)                
            else:
                None      
        if dest not in chemin:
            chemin = None           
        return chemin

    

    def connected_components(self):
        list_components = []
        marked_sommet = {sommet:False for sommet in self.nodes} #O(n): complexite a peu pres le nb de noeuds dans le graphe 
        def dfs(sommet):
            component = [sommet]
            for neighbour in self.graph[sommet]:
                neighbour = neighbour[0]
                if not marked_sommet[neighbour]: #O(n+m): visite chq arrete et chq sommet une fois 
                    marked_sommet[neighbour] = True
                    component += dfs(neighbour)
            return component

        for sommet in self.nodes:
            if not marked_sommet[sommet]:
                list_components.append(dfs(sommet))
                
        return list_components
    
    



    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        raise NotImplementedError


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

    f=open(filename, "r")
    g=Graph()
    line1= f.readline()
    list_line1=line1.split()
    g=Graph([node for node in range(1, int(list_line1[0])+1)]) #on crée le graphe avec le bon nb de noeuds
    for i in range(1, int(list_line1[1])+1):
        linei=f.readline()
        list_linei=linei.split()
        if len(list_linei)!=4:
            g.add_edge(int(list_linei[0]), int(list_linei[1]), int(list_linei[2]))
        else:
            g.add_edge(int(list_linei[0]), int(list_linei[1]), int(list_linei[2]), int(list_linei[3]))
    return g

"""
def connected_components(filename):
    g=graph_from_file(filename)
    s=sommet_depart=#a completer i guess
    list_components=[]
    list_pile=[]
    list_pile.append() #rajouter un element a la fin
    list_pile.pop() #recuperer le dernier element 
    list_pile=[#recuperer le 1er node]
    if list_pile!=[]:
        node=list_pile.pop()
        list_components.append(node)
        list_arretes=g[node]
        for i in range(1,len(list_arretes)+1):
            list_pile.append(list_arretes[i][0])"""

        













    