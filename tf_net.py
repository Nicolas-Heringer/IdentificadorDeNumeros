import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_treino, y_treino), (x_teste,y_teste) = mnist.load_data()

print(y_treino.shape)
print(y_teste.shape)

x_treino = x_treino.reshape(-1,28*28).astype("float32")/255 # Transforma em um vetorzão normalizado
x_teste = x_teste.reshape(-1,28*28).astype("float32")/255 

# Sequential API (keras)
# 1 input -> 1 output

model = keras.Sequential(
	[
		keras.Input(shape=(28*28)),
		layers.Dense(512, activation="relu"),
		layers.Dense(256, activation="relu"),
		layers.Dense(10, activation="sigmoid")
	]
)

print(model.summary())

model.compile(
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), # from_logits=True faz 'softmax' antes
	optimizer = keras.optimizers.Adam(lr=0.001),
	metrics = ["accuracy"]
)

model.fit(x_treino, y_treino, batch_size=30, epochs=100, verbose=1)

model.save('meu_modelo.h5')