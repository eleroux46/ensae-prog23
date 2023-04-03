from graph import stock_results, graph_from_file
import sys 
sys.path.append("delivery_network")

import unittest   # The test framework

class Test_Results(unittest.TestCase):
    def test_routes1(self):
        g = graph_from_file("input/network.1.in")
        file1 = open("input/routes.1.in", "r")
        file2 = open("output/routes.1.out", "r")
        file1.readline()
        line1= file1.readline().split()
        src, dest, utilite = int(line1[0]), int(line1[1]), line1[2]
        power = g.min_power_kruskal(src, dest) 
        line1bis= file2.readline()
        self.assertEqual(line1bis, str(power)+" "+str(utilite)+"\n")

if __name__ == '__main__':
    unittest.main()