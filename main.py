########################################################
# IMPORTATIONS
########################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


########################################################
# FONCTION TO APPROXIMATE
########################################################

def complex_function(x):
    return 3 * x * x + 2 * x + 6


########################################################
# CREATE SAMPLES
########################################################

# Generation of X, Y and xTest
xData = np.linspace(-np.pi, np.pi, 1000)
yData = complex_function(xData)
xTest = np.linspace(-np.pi, np.pi, 100)


########################################################
# SIMPLE MODEL CREATION
########################################################

# Creation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')


########################################################
# CALL BACK FOR PLOT
########################################################

# Callback for prediction history
class PredictionHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.predictions_per_epoch = []

    def on_epoch_end(self, epoch, logs=None):
        self.predictions_per_epoch.append(model.predict(xTest).flatten())

# Callback for weight history
class WeightsHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.weightsPerEpoch = []

    def on_epoch_end(self, epoch, logs=None):
        weights = [layer.get_weights()[0] for layer in model.layers]
        self.weightsPerEpoch.append(weights)

# call back creation
prediction_history = PredictionHistory()
weights_history = WeightsHistory()


########################################################
# FIT THE MODEL
########################################################

history = model.fit(xData, yData, epochs=50, callbacks=[prediction_history, weights_history])



########################################################
# SHOW APPROXIMATION 
########################################################

def update_plot(i, x, y, line):
    line.set_data(x, prediction_history.predictions_per_epoch[i])
    return line,

fig, ax = plt.subplots()
x, y = xTest, prediction_history.predictions_per_epoch[0]
line, = ax.plot(x, y, 'r-')
ax.plot(xData, yData, 'b-', label='Fonction originale')
plt.legend()

# Créer l'animation des prédictions
ani = FuncAnimation(fig, update_plot, frames=len(prediction_history.predictions_per_epoch),
                    fargs=(x, y, line), blit=True)

plt.show()


########################################################
# SHOW LOSS GRAPHIC
########################################################

plt.figure()
plt.plot(history.history['loss'], label='Loss')
plt.title('Graphique de la perte durant l\'entraînement')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.show()



########################################################
# SHOW HEATMAP WEIGHT / LAYER 1
########################################################

def update_heatmap(epoch, data, ax):
    ax.clear()
    #FIRST LAYER
    layer_weights = data[epoch][0]  
    ax.imshow(layer_weights, cmap='hot', aspect='auto')
    ax.set_title(f'Epoch: {epoch + 1}')



# Créer la figure pour l'animation de heatmap
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_heatmap, frames=50,
                    fargs=(weights_history.weightsPerEpoch, ax))

plt.show()

########################################################
# SHOW HEATMAP WEIGHT / LAYER 2
########################################################

def update_heatmap(epoch, data, ax):
    ax.clear()
    #SECOND LAYER
    layer_weights = data[epoch][1]  
    ax.imshow(layer_weights, cmap='hot', aspect='auto')
    ax.set_title(f'Epoch: {epoch + 1}')


# Créer la figure pour l'animation de heatmap
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_heatmap, frames=50,
                    fargs=(weights_history.weightsPerEpoch, ax))

plt.show()



