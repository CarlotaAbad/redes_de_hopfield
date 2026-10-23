# Red de Hopfield para clasificar dígitos MNIST

# María Carlota Abad Esteban y Adriana Soria Franco

## Qué hace
Implementa una red de Hopfield que guarda 70 dígitos de MNIST con la regla de Hebb
y clasifica otros 30. Incluye la dinámica síncrona y asíncrona, la función de energía,
pruebas con ruido (5 %, 10 % y 20 %), un estudio de la capacidad y la regla de proyección.

## Archivos
- `hopfield_mnist.py`: el código
- `mnist/`: los ficheros originales de MNIST
- `figuras/`: las gráficas que genera el código
- `resultados.json` y `salida_consola.txt`: los resultados

## Cómo ejecutarlo
    pip install numpy matplotlib scikit-learn
    python hopfield_mnist.py
