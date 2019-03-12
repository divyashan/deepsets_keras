import keras
from keras.layers import Dense, Flatten, Input, Add
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
def create_phi(input_dim):
  x0 = Input(shape=input_dim, name='Input')

  x = Dense(2)(x0)
  x = Flatten()(x)
  y = Dense( 2, name='dense_encoding', activation='relu' )(x)

  model = Model( inputs = x0, outputs = y)
  return model

def create_rho( input_dim, phi_model, n_inputs ):
	inputs = [Input( input_dim )for i in range(n_inputs)]
	outputs = [phi_model(i) for i in inputs]

	added_rep = Add()(outputs)

	output = Dense(1, activation='relu')(added_rep)
	model = Model(inputs, output, name='siamese')
	return model


def generate_data(n_sets=50, set_dims=(3,1)):
  all_x = []
  all_y = []
  bernoulli_params = np.random.rand(n_sets, 1)
  
  for s in range(n_sets):
    # pick param for bernoulli
    param = bernoulli_params[s] 
    x_vals = np.random.binomial(1, param, set_dims)
    y_vals = np.sum(x_vals)
    all_x.append(x_vals.T)
    all_y.append(y_vals)
  all_x = np.array(all_x)
  all_y = np.array(all_y)
  
  all_x = np.expand_dims(all_x, 2)
  all_x = np.moveaxis(all_x,3, 0)
  all_x = [x for x in all_x]
  return all_x, all_y





