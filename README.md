# Classificador de Cães e Gatos com TensorFlow 🐶🐱

Este projeto implementa um modelo de classificação binária de imagens de **Cães vs Gatos** usando o **TensorFlow** e o dataset `cats_vs_dogs` do **TensorFlow Datasets**. O objetivo é treinar um modelo de rede neural convolucional (CNN) para identificar imagens de cães e gatos.

## 🚀 Principais Características
- **Dataset Cats vs Dogs**: Utiliza o famoso dataset de imagens de cães e gatos, com um total de 25.000 imagens.
- **Aumento de Dados (Data Augmentation)**: Aumento de dados para melhorar a generalização do modelo, incluindo flip horizontal/vertical, variação de brilho, contraste e saturação.
- **Modelo CNN**: Rede neural convolucional com múltiplas camadas para extração de características e classificação de imagens.
- **Early Stopping e ModelCheckpoint**: Uso de callbacks para interromper o treinamento automaticamente se a perda de validação não melhorar e salvar o melhor modelo.

## 🎯 Objetivo
O objetivo deste projeto é demonstrar a construção de um modelo CNN para classificação binária, utilizando boas práticas de pré-processamento de dados, otimização de performance e técnicas de regularização para evitar overfitting.


## 🧑‍💻 Tecnologias Utilizadas
- TensorFlow para treinamento do modelo de redes neurais.
- Keras para construção da arquitetura da CNN.
- TensorFlow Datasets para carregamento do dataset cats_vs_dogs.
