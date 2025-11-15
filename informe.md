# Models
## SIMPLE RNN
### 1.2 SIMPLE RNN con packing sequence

En la exploración de datos notamos que éstos contienen una gran cantidad de ceros al final. En una RNN simple, esto provoca que la red “se olvide” de la información inicial (que en este caso es la más relevante) y se concentre en los elementos finales de la secuencia, que en este caso son ceros sin contenido útil. Como resultado, el modelo aprende patrones irrelevantes y su desempeño empeora.

En PyTorch tenemos la función pack_padded_sequence, que nos permite manejar secuencias con padding. Esta función permite que la RNN procese únicamente las partes no rellenas de cada secuencia, evitando que el modelo gaste recursos en procesar los ceros añadidos artificialmente.

En nuestro caso, las secuencias ya incluyen ceros al final, por lo que podemos tomarlas como si les hubieramos aplicado padding. Debido a esto, solo es necesario utilizar el mecanismo de packing para que la RNN ignore correctamente el padding y se concentre en la información útil.

La idea es probar si esto mejora el desempeño del modelo.

Referencias: 

https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html

https://www.geeksforgeeks.org/deep-learning/how-do-you-handle-sequence-padding-and-packing-in-pytorch-for-rnns/

# Training
## Training RNN
### Observaciones RNN SIMPLE
De todo lo que hicimos NADA la hizo funcionar bien, y menos en comparación con GRU y LSTM.

Por alguna razon, dio mejor borrar 75 cerros "a mano" que usar packing. Estará bien implementado el packing???

RNN borrando ceros a mano:

Accuracy: 0.4307

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.95      0.38      0.54     18118
           1       0.09      0.54      0.15       556
           2       0.18      0.47      0.26      1448
           3       0.03      0.88      0.05       162
           4       0.59      0.88      0.70      1608

    accuracy                           0.43     21892
   macro avg       0.37      0.63      0.34     21892
weighted avg       0.84      0.43      0.52     21892

RNN con packing:

Accuracy: 0.3335

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.92      0.28      0.43     18118
           1       0.07      0.16      0.09       556
           2       0.22      0.45      0.29      1448
           3       0.02      0.83      0.04       162
           4       0.24      0.81      0.37      1608

    accuracy                           0.33     21892
   macro avg       0.29      0.51      0.25     21892
weighted avg       0.80      0.33      0.41     21892

## Training GRU
### Observaciones GRU
GRU:

Accuracy: 0.9818

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99     18118
           1       0.85      0.81      0.83       556
           2       0.93      0.95      0.94      1448
           3       0.77      0.77      0.77       162
           4       0.99      0.98      0.99      1608

    accuracy                           0.98     21892
   macro avg       0.91      0.90      0.90     21892
weighted avg       0.98      0.98      0.98     21892

GRU con sampler:

Accuracy: 0.9637

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.97      0.98     18118
           1       0.57      0.84      0.68       556
           2       0.87      0.96      0.91      1448
           3       0.61      0.88      0.72       162
           4       0.98      0.98      0.98      1608

    accuracy                           0.96     21892
   macro avg       0.81      0.92      0.86     21892
weighted avg       0.97      0.96      0.97     21892

La GRU con sampler no dio mal, pero empeoró versus la GRU "común".

Z-score y augmentation, parecido a usar sampler.

GRU bidireccional:

Accuracy: 0.9863

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      1.00      0.99     18118
           1       0.90      0.83      0.86       556
           2       0.96      0.96      0.96      1448
           3       0.87      0.76      0.81       162
           4       0.99      0.98      0.99      1608

    accuracy                           0.99     21892
   macro avg       0.94      0.91      0.92     21892
weighted avg       0.99      0.99      0.99     21892

La bidireccional da mucho mejor en accuracy que la GRU simple, que es la mejor que teníamos hasta ahora. Mejora también en las demás métricas pero no es tan notable la diferencia. Si se nota bastante mejora en las 2 clases que venian dando peor, que son la 1 y la 3.

## Training LSTM

Accuracy: 0.9646

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.97      0.99      0.98     18118
           1       0.93      0.55      0.69       556
           2       0.93      0.87      0.90      1448
           3       0.74      0.42      0.54       162
           4       0.94      0.92      0.93      1608

    accuracy                           0.96     21892
   macro avg       0.90      0.75      0.81     21892
weighted avg       0.96      0.96      0.96     21892

En comparacion con la GRU "comun" y la bidireccional, dio peor.

LSTM bidireccional:

Accuracy: 0.9836

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99     18118
           1       0.88      0.77      0.82       556
           2       0.96      0.96      0.96      1448
           3       0.81      0.80      0.81       162
           4       0.99      0.99      0.99      1608

    accuracy                           0.98     21892
   macro avg       0.93      0.90      0.91     21892
weighted avg       0.98      0.98      0.98     21892


Mejora notoria vs la LSTM "comun". Comparable a la performance de la GRU bidireccional pero aun un poco por debajo.

# Weight and Biases

## Análisis GRU W and B
| **Modelo**            | **hidden_dim** | **n_layers** | **Bidireccional** | **LR**    | **Accuracy** | **Macro F1** | **Observaciones**                                                                 |
| :-------------------- | :------------- | :----------- | :---------------- | :-------- | :----------- | :----------- | :-------------------------------------------------------------------------------- |
| GRU (W&B – mejor run) | 128            | 1            | No                | 0.0064    | 0.980        | 0.89         | Alto rendimiento general, pero menor capacidad para clases minoritarias.          |
| GRU (W&B run #2)      | 32             | 3            | No                | 0.0117    | 0.974        | 0.85         | Mayor profundidad no introdece mejoras, esto puede estar contra-arrestado para el hecho de tener solo 32dim en la capa ocultas|
| GRU (W&B run #3)      | 128            | 1            | No                | 0.0098    | 0.973        | 0.85         | Opuesto al caso anterior, mayor capa ocualta pero sin suficiente profundidad |
| GRU (W&B run #4)      | 128            | 1            | No                | 0.0449    | 0.827        | 0.18         | Entrenamiento inestable generado seguramente por la tasa de aprendizaje alta.                             |
| GRU (W&B run #5)      | 32             | 1            | No                | 0.0266    | 0.934        | 0.56         | Arquitectura demasiado chica, con baja profundidas y una baja dimension de capa oculta|
| **GRU (original)**    | **64**         | **2**        | **Sí**            | **0.001** | **≈ 0.98**   | **0.92**     | Mejor equilibrio entre precisión y recall; detecta mejor las clases minoritarias. |

Todas las arquitecturas probadas en W&B son más pequeñas que la GRU original, ya sea por tener una menor profundidad (n_layers) o un hidden dim más reducido.
Ninguna de estas configuraciones logró superar los resultados obtenidos por nuestra arquitectura base.

Dado que los experimentos de W&B no incluyeron modelos más grandes o complejos, no es posible afirmar que nuestra arquitectura sea la mejor.
Sin embargo, los resultados muestran que la GRU original genera un excelente equilibrio entre capacidad de representación y rendimiento, motivo por el cual decidimos mantenerla como la configuración final.


## Análisis W&B LSTM

| **Modelo**            | **hidden_dim** | **n_layers** | **LR** | **Accuracy** | **Macro F1** | **Observaciones**                                                                          |
| :-------------------------------- | :------------- | :----------- | :----- | :----------- | :----------- |  :----------- |
| **LSTM (W&B run #1)** | 128            | 3            | 0.0079 | 0.907        | 0.49         | Aumento en el tamaño de la arquitectura genera peores resultados   |
| **LSTM (W&B run #2)** | 128            | 1            | 0.0029 | 0.983        | 0.90         | Balance entre hideen dims y profundida genera un resultado tan bueno como el de la arquitectura base |
| **LSTM (W&B run #3)** | 32             | 3            | 0.0065 | 0.969        | 0.84         | Aumentar la profundidad sigue geerando resultados por debajo de lo esperado aunque se haya intentendo balancear con un menor hidden dim |
| **LSTM (W&B run #4)** | 128            | 3            | 0.0033 | 0.970        | 0.86         | Buen desempeño global; *label smoothing* ayudó a suavizar el aprendizaje, pero sin superar al modelo base. |
| **LSTM (W&B run #5)** | 32             | 1            | 0.0075 | 0.979        | 0.89         | Modelo liviano, estable y con resultados cercanos al mejor run, aunque con menor capacidad representativa. |
| **LSTM (original)**   | 64             | 2            | 0.001  | ≈0.984       | 0.91         | Arquitectura equilibrada; destaca por su estabilidad y mejor detección de clases minoritarias.             |


Los resultados obtenidos en W&B muestran que las distintas configuraciones de LSTM alcanzan rendimientos altos, pero ninguna logra superar a la arquitectura original.
Las variantes más simples (con menor número de capas o hidden_dim reducido) mantienen una buena accuracy, aunque con un menor macro F1, lo que indica un desempeño más limitado en las clases minoritarias.

Se observa además una tendencia en la que las arquitecturas más grandes, especialmente las de mayor profundidad, obtienen resultados por debajo de lo esperado y presentan un costo computacional más alto.

La LSTM original alcanza el mejor equilibrio gracias a una configuración intermedia entre hidden_dim y n_layers, que ofrece buena capacidad de representación sin caer en sobreajuste.
Por este motivo, se mantiene como la arquitectura final seleccionada.