from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Завантаження датасету
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Збереження перших 10 зображень
for i in range(10):
    img = Image.fromarray(x_test[i])
    img = img.convert("L")  # grayscale
    img.save(f"example_{i}.png")
