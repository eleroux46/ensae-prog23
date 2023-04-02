from graph import Graph, graph_from_file, UnionFind, compare, estimate_time, stock_results


data_path = "input/"
file_name = "network.1.in"

g = graph_from_file(data_path + file_name)
h=g.kruskal()
print(h)
h.graph[1]
#compare(2)
#estimate_time(2)
stock_results(2)
#print(g.build_caracteristics())
#print(h.list_rank, h.list_parent)
#print(g.connected_components())
print(g.min_power_kruskal(10,18))
# print(g.get_path_with_kruskal(15, 18))
#print(h.min_power(10, 10))
#print(val=h.graph.values())
#print(val[0])

