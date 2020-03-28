import numpy as np


class neural_network:
      #the constructor of input and output training set
      def __init__(self,x,y):
                self.x=x
                self.y=y

      # the sigmoid function
      def sigmoid(self,out):
                return 1 / ( 1 + np.exp(-out))


     # the training inputs
      def training_inputs(self):
                return  np.array(self.x)

     # the training output
      def training_outputs(self):
                return np.array(self.y).T

     # train the model
      def generating_model(self,t_input):
                np.random.seed(1)
                random_weights = 2 * np.random.random((3, 1))-1
                for iterx in range(1):
                        output = np.dot(t_input, random_weights)
                        activiation_output = self.sigmoid(output)
                        print(activiation_output)

#the main

model = neural_network ([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]],[[0, 1, 1, 0]])

t_input= model.training_inputs()
t_output= model.training_outputs()
model.generating_model(t_input)
