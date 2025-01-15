import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

rows = 10
cols = 10

num_of_samples = []
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols, rows))
fig.tight_layout()
for i in range(cols):
    for j in range(rows):
        images = x_train[y_train == j]
        axs[j][i].imshow(images[random.randint(0, len(images - 1)), :, :],
                         cmap=plt.get_cmap("gray"))

plt.show()
