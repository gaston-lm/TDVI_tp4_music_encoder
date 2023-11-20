---
title: "Trabajo Práctico 4 - Encoders, decoders y autoencoders"
subtitle: "Tecnología Digital VI: Inteligencia Artificial"
author: [Federico Giorgi,Gastón Loza Montaña,Tomás Curzio]
date: "24/11/23"
geometry: "left=3cm,right=3cm,top=2.5cm,bottom=2.5cm"
lang: "es"
...

# Encodear canciones en vectores latentes

Para este primer inciso, utilizamos una red sencilla, muy similar a los autoencoders vistos en clase, utilizando la estructura que se puede ver en la figura 1.

![Estructura autoencoder](estructura.png)


En un principio, pudimos notar que la estrategia de hacer muchos canales en un inicio y luego disminuirlos no nos dió buenos resultados. Por ello, probamos ir disminuyendo de a poco o mantener los canales a lo largo de la red, para finalmente flattenear el vector y obtener así nuestro vector latente. Realizamos esto con distintos parámetros que nos dejaron vectores latentes de distintos tamaños, hasta llegar al punto de que la canción sea prácticamente irreconocible. Los audios resultantes se pueden ver en la siguiente tabla, que utiliza Music - Maddona como ejemplo.

¿Quizás agregar a la tabla los parámetros? (in/out channels, kernel size, stride, etc)

|    Vector latente     |   |
|:----------------------|--:|
| 1x110250 (Original)   | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/13dINLPdNFMA_Rz2TYi6Tr5xfH0_xIw-c/view) |
| 1x55112               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1cnege9wbLe6OsE1E8LDJliZn1zqfrBRD/view) |
| 1x32151               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1jql5XVD3KKCA10C2dFdnZXvLSiJ9QSPX/view) |
| 1x24496               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1UY3wKQtwLF5gk3JVnVQCwwSjJQL_2Dgl/view) |
| 1x18376               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1he1mJwi92Qtt95sVPn0LRAQI-YC0MCfH/view) |
| 1x9184                | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1-8SM5pMs8vOZSLXxOH3wCtGWIl2ycoSL/view) |

En cuanto a los otros hiperparámetros, tomamos las siguientes decisiones.

- Decidimos dejar las epochs que venían por defecto en la notebook otorgada, por un tema de tiempo de cómputo, además de que observamos que la loss comenzaba a mantenerse bastante estable y no parecía tener mucha proyección a disminuir en futuras epochs.
- En cuanto al learning rate, corrimos un hyperopt con algunas iteraciones y el resultado obtenido (aprox 0.015) fue muy cercano al learning rate que venía por defecto en la notebook otorgada. Corrimos algunos experimentos con el learning rate obtenido y la diferencia era prácticamente nula, entonces a fines de poder comparar entre los distintos experimentos que ya habíamos realizado, decidimos mantener este parámetro en 0.02.
- No se me ocurre acá como explicar bien porque mantuvimos el batch size como venía.

Para no tener solamente una diferencia de audio como referencia, también observamos como difieren tanto la waveform como el espectograma de la canción original, con los de los distintos audios encodeados y luego decodeados, observados en la tabla. Estos fueron los resultados:

Acá pondría las imagenes de los waveforms y de los espectogramas, primero el del original y despues el de los demás en orden decreciente.

Observando y escuchando, decidimos quedarnos como "vector de mínimo tamaño" el de ¿1x18376? ¿1x24496?, ya que consideramos que mantiene una similitud razonable con el audio original, a diferencia de el vector de 1x9184, que es prácticamente inentendible. Con esto, ya podemos avanzar a un análisis exploratorio de los vectores latentes obtenidos.

# Análisis exploratorio de vectores latentes

Aca tonga la tenés mucho mas clara vos, igual mañana lo charlamos pero no quería dejar de redactar al menos un poco de lo q pense del punto 3 para tener al menos la idea.

# Encodeo de música nueva

Con la red definida, en un principio no se podría encodear música nueva, a no ser que tanto su sample rate como su tamaño original, sea el mismo que el de los audios del dataset que estamos utilizando para entrenar y validar nuestra red.

Para poder hacerlo, hay algunas opciones:

1. Adaptar el audio que querramos encodear para cumplir los requisitos de la red.
2. Adaptar la red para el sample rate y tamaño original de aquello que querramos encodear.
3. Adaptar el sample rate que viene por defecto al del audio que querramos encodear, y reducir el tamaño del mismo a los requisitos de la red (recortandolo a 5s como los demás).

Nosotros realizamos ??? y pudimos lograr encodear música nueva. A modo de ejemplo, podemos ver los audios obtenidos a partir de la canción ????

¿Quizás agregar a la tabla los parámetros? (in/out channels, kernel size, stride, etc)

|    Vector latente     |   |
|:----------------------|--:|
| 1x110250 (Original)   | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/13dINLPdNFMA_Rz2TYi6Tr5xfH0_xIw-c/view) |
| 1x55112               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1cnege9wbLe6OsE1E8LDJliZn1zqfrBRD/view) |
| 1x32151               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1jql5XVD3KKCA10C2dFdnZXvLSiJ9QSPX/view) |
| 1x24496               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1UY3wKQtwLF5gk3JVnVQCwwSjJQL_2Dgl/view) |
| 1x18376               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1he1mJwi92Qtt95sVPn0LRAQI-YC0MCfH/view) |
| 1x9184                | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1-8SM5pMs8vOZSLXxOH3wCtGWIl2ycoSL/view) |

# Generación de música nueva.

Hemos visto en clase como este tipo de autoencoders que hemos programado, no son lo mejor para generar algo nuevo. Es por eso, que han surgido otros modelos, como pueden ser VAE o GAN. Sin embargo, se pueden probar cosas como generar vectores random del tamaño del espacio latente y pasarlos por el decoder o incluso promediar vectores latentes de otras canciones y decodearlos. 

Este tipo de cosas son las que probamos hacer y, si bien lo obtenido no es lo mas satisfactorio, logramos generar audio completamente nuevo a partir de nuestra red. Se pueden escuchar los audios obtenidos en la siguiente tabla:

|    Vector latente     |   |
|:----------------------|--:|
| 1x110250 (Original)   | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/13dINLPdNFMA_Rz2TYi6Tr5xfH0_xIw-c/view) |
| 1x55112               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1cnege9wbLe6OsE1E8LDJliZn1zqfrBRD/view) |
| 1x32151               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1jql5XVD3KKCA10C2dFdnZXvLSiJ9QSPX/view) |
| 1x24496               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1UY3wKQtwLF5gk3JVnVQCwwSjJQL_2Dgl/view) |
| 1x18376               | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1he1mJwi92Qtt95sVPn0LRAQI-YC0MCfH/view) |
| 1x9184                | [$\textcolor{blue}{link}$](https://drive.google.com/file/d/1-8SM5pMs8vOZSLXxOH3wCtGWIl2ycoSL/view) |

(copie siempre la misma tabla por las dudas, me parece una buena manera de mostrar los resultados pero lo podemos cambiar si se nos ocurre algo mejor)