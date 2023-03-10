# À compléter
import sys 
sys.path.append("delivery_network/")

import unittest 
from graph import Graph, graph_from_file

class Test_mindist(unittest.TestCase):
    def testnetwork0(self):
        g = graph_from_file("input/network.04.in")



if __name__ == '__main__':
    unittest.main()
