import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Carregar o dataset cats_vs_dogs do TensorFlow Datasets
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Separar o dataset em treino (train) e validar com uma divisão
train_dataset = dataset['train']

# Obter o tamanho do dataset
dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()

# Dividir o conjunto de treino em treino e validação (80% treino, 20% validação)
train_size = int(0.8 * dataset_size)
validation_size = dataset_size - train_size

# Criar os splits de treino e validação
train_dataset = train_dataset.take(train_size)
validation_dataset = dataset['train'].skip(train_size)

# Função de pré-processamento para aumentar e normalizar as imagens
def preprocess_image(image, label):
    # Redimensionar para 150x150 (tamanho de entrada do modelo)
    image = tf.image.resize(image, (300, 300))
    
    # Aumento de dados: Flip horizontal, brilho, contraste, saturação e matiz
    image = tf.image.random_flip_left_right(image)  # Flip horizontal
    image = tf.image.random_flip_up_down(image)  # Flip vertical
    image = tf.image.random_brightness(image, max_delta=0.2)  # Alterar brilho
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # Contraste aleatório
    image = tf.image.random_hue(image, max_delta=0.2)  # Alterar matiz
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # Saturação aleatória
    
    # Normalizar os pixels para o intervalo [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Aplicar o pré-processamento ao dataset de treino, validação e teste
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Ajustar o número de batchs e otimizar o desempenho
train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Construir o modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Saída binária: cão ou gato
])

# Compilar o modelo
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Definir um callback para parar o treinamento mais cedo se a acurácia não melhorar
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    #callbacks=[early_stopping]
)

# Salvar o modelo treinado
model.save('dog_cat_classifier.h5')

# Avaliar o modelo (como o dataset de teste não está disponível, avaliaremos no conjunto de validação)
loss, accuracy = model.evaluate(validation_dataset)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

