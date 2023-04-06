from graph import Graph, graph_from_file, UnionFind, compare, estimate_time, stock_results, Catalogue, catalogue_from_file, glouton_algorithm, estimate_time_kruskal, Simulated_annealing_glouton, Simulated_annealing_random


data_path = "input/"
file_name = "network.1.in"

g = graph_from_file(data_path + file_name)
h=g.kruskal()
print(h)
h.graph[1]
#compare(2)
#estimate_time_kruskal(2)
#stock_results(10)
#print(g.build_caracteristics())
#print(h.list_rank, h.list_parent)
#print(g.connected_components())
#print(g.min_power_kruskal(6,11))
# print(g.get_path_with_kruskal(15, 18))
#print(h.min_power(10, 10))
#print(val=h.graph.values())
#print(val[0])

#print(catalogue_from_file(1))
#glouton_algorithm(2,2)
#Simulated_annealing_glouton(2,2)
Simulated_annealing_random(2,2)
#print(backpack_algorithm(5,2))
