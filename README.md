# **BPETokenizer**
Una implementación didáctica en Python del algoritmo Byte-Pair Encoding (BPE).

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg) ![alt text](https://img.shields.io/badge/python-3.8+-blue.svg)

Este mini proyecto es la implementación del algoritmo Byte-Pair Encoding (BPE) para tokenizar texto, tal y como lo hacen los tokenizadores de la serie de Modelos de Lenguaje (LLMs) como GPT, Llama o Mistral.

## **Índice**
* Propósito y Advertencia
* Instalación
* ¿Cómo Usarlo?
    1. Preparar el Corpus
    2. Entrenar el Tokenizador
    3. Guardar y Cargar
    4. Tokenizar y De-tokenizar Texto
* ¿Cómo Funciona el Algoritmo BPE?
* Licencia
* Agradecimientos
* Bibliografía


## ⚠️**Propósito y Advertencia**⚠️
Este proyecto es una implementación sencilla del algoritmo BPE en Python que, pese a permitir el entrenamiento e inferencia de tokenizadores, no debería utilizarse en entornos de producción. Este proyecto tiene como objetivo comprender el algoritmo de Byte-Pair Encoding y el funcionamiento de los tokenizadores, si lo que buscas es utilizar tokenizadores de manera eficiente puedes ver los siguientes proyectos:
* [`tiktoken`](https://github.com/openai/tiktoken)
* [`sentencepiece`](https://github.com/google/sentencepiece)
* [`tokenizers`](https://github.com/huggingface/tokenizers)

## ****