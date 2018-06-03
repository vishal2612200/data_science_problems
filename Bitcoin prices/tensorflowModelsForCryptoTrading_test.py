# Libraries
import unitest
# Local module
from tensorflowModelsForCryptotrading import *

# TODO: Complete unittests

class VanillaNeuralNetwork_test(unitest.TestCase):
	def setUp(self):
		self.nn = VanillaNeuralNetwork()

	def tearDown(self):
		pass

	def test_assertWeightInit(self):
		pass
		# self.assertEquals()

	def test_modelIsLearning(self):
		x_data = np.array(np.arange(-3,3,0.1)).reshape(1, -1)
		y_data = np.array([np.sin(i) for i in x_data]).reshape(1, -1)
		nn = VanillaNeuralNetwork(x = x_data,
															y = y_data,
															layers = 3,
															neurons = [1, 20, 1],
															learningRate = 0.001,
															epochs = 1000)

		nn.train()

if __name__ == "__main__":
	unitest.main()
	