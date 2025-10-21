import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from typing import cast
from keras.models import load_model, Model

# Cargar el modelo h5
modelo_h5 = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn/models/riesgo_3conv.h5'
riesgo_model = cast(Model, load_model(modelo_h5))

images = []
# AQUI ESPECIFICAMOS UNAS IMAGENES
filenames = ['/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn/test_images/PruebaFormula1.jpg']

for filepath in filenames:
    image = plt.imread(filepath)
    image_resized = resize(image, (21, 28), anti_aliasing=True, clip=False, preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8)  # Convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.

predicted_classes = riesgo_model.predict(test_X)

# Asegúrate de tener una lista de etiquetas o categorías en 'sriesgos'
sriesgos = ['golf', 'basket', 'tenis', 'natacion', 'ciclismo', 'beisball', 'futbol', 'americano', 'f1', 'boxeo']

for i, img_tagged in enumerate(predicted_classes):
    print(filenames[i], sriesgos[np.argmax(img_tagged)])
