# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import graph_from_file
import unittest   # The test framework

class Test_MinimalPower(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        self.assertEqual(g.min_power(1, 4)[1], 11)
        self.assertEqual(g.min_power(2, 4)[1], 10)

    def test_network1(self):
        g = graph_from_file("input/network.1.in")
        self.assertEqual(g.min_power(2, 4)[1], 13)
        self.assertEqual(g.min_power(20, 11), ([20, 8, 11], 11))

    def test_network03(self):
        g = graph_from_file("input/network.03.in")
        self.assertEqual(g.min_power(2, 5), None)
        

if __name__ == '__main__':
    unittest.main()
