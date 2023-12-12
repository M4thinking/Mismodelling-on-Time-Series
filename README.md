Para correr el código:

1. Clonar el repositorio
2. Instalar las dependencias del requirements.txt
3. Poner los edf de la página de zenodo para los eeg en la carpeta data externa a modd
4. Correr datasets.py para generar los datos en segmentos, filtrados, normalizados y listos para entrenar.
5. Correr train_transformer.py para entrenar el modelo
6. Una vez entrenado se genera un checkpoint en la carpeta de tst_logs externa a modd.
7. Si está satisfecho con el modelo, pasar a la carpeta interna de modd y correr eval_model.py para generar las predicciones en la carpeta tst_logs interna a modd, si no quiere entrenar, simplemente correr eval_model.py y se cargará el checkpoint de la carpeta tst_logs interna a modd.
8. Cambiar carpeta de TSP-IT-Replace por la carpeta del repositorio de TSP-IT compilado, El link del respotirio se encuentra en "MI Estimator/utils-and-example".
9. Correr mi_estimator.py para generar las predicciones, con su respectiva matriz de información mutua e información mutua promedio, entre otras métricas y gráficos.