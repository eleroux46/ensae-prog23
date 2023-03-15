# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import graph_from_file, estimate_time, UnionFind
import unittest   # The test framework

class Test_MinimalPower(unittest.TestCase):
    def test_network02(self):
        g = graph_from_file("input/network.02.in")
        mst=g.kruskal()
        self.assertEqual(mst.graph[1], [(4,4,1)])
        self.assertEqual(mst.graph[2], [(3,4,1)])

    def test_network03(self):
        g = graph_from_file("input/network.03.in")
        mst=g.kruskal()
        self.assertEqual(mst.graph[3], [(2,4,1), (4,4,1)])
        self.assertEqual(mst.graph[4], [(3,4,1)])

if __name__ == '__main__':
    unittest.main()
