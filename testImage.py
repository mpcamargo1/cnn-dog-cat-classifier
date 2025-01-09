import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Carregar o modelo treinado
model = tf.keras.models.load_model('dog_cat_classifier.h5')

print(model.summary())

# Função para prever a classe de uma imagem
def predict_image(image_path):
    # Carregar a imagem e redimensionar para o tamanho usado no treinamento (300x300))
    image = load_img(image_path, target_size=(300, 300))
    
    # Converter a imagem em um array numpy e normalizar os valores de pixel
    image_array = img_to_array(image) / 255.0  # Normalização para [0, 1]
    
    # Adicionar uma dimensão para simular um batch (modelo espera uma entrada em forma de batch)
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Fazer a previsão
    prediction = model.predict(image_batch)
    
    # Interpretar o resultado
    if prediction[0] > 0.5:
        return "Dog", prediction[0][0]  # Se a previsão for maior que 0.5, é um cachorro
    else:
        return "Cat", prediction[0][0]  # Caso contrário, é um gato

# Caminho da imagem a ser testada
image_path = 'dog.jpg'  # Substitua pelo caminho da sua imagem

# Fazer a previsão
label, confidence = predict_image(image_path)
print(f"Predicted label: {label} with confidence: {confidence:.2f}")

