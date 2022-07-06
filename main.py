from pickletools import optimize
from tkinter import *
import math
from random import *
import pandas as pd

# TensorFlow valoresDeSalida tf.keras
import tensorflow as tf

# Librerias de ayuda
import numpy as np
from matplotlib import pyplot, units


def Graficar(Iteraciones, Resultados):
    fig, grafica = pyplot.subplots()
    grafica.set_title("Grafica")
    grafica.plot(Iteraciones, Resultados)
    grafica.set_xlabel("Iteraciones")
    grafica.set_ylabel("valoresDeSalida")
    pyplot.savefig("Grafica")
    pyplot.close()


def AlgebraLinealSolve(eta, iteraciones):
    Iteraciones = []
    PesosDeEntradaALaNeurona = []
    ResultadosObtenidosPorIteracion = []
    for i in range(4):
        PesosDeEntradaALaNeurona.append(uniform(0, 1))
    print('Valores aleatorios: ', PesosDeEntradaALaNeurona)
    # leyendo información del archivo csv
    DatosDeEntradaCSV = pd.read_csv('193259.csv', header=None, skiprows=1, names=[
        'x1', 'x2', 'x3', 'valoresDeSalida'], usecols=['x1', 'x2', 'x3'])
    DatosDeResultadosCSV = pd.read_csv('193259.csv', header=None, skiprows=1,
                                       names=['x1', 'x2', 'x3', 'valoresDeSalida'], usecols=['valoresDeSalida'])
    ValorDeEta = []
    ValorDeEta.append(float(eta))
    # Creando matrices
    DatosDeEntradaCSV = np.matrix(DatosDeEntradaCSV)
    DatosDeEntradaCSV = np.c_[np.ones(100), DatosDeEntradaCSV]
    ValorDeEta = np.matrix(ValorDeEta)
    DatosDeResultadosCSV = np.matrix(DatosDeResultadosCSV)

    for iteracion in range(int(iteraciones)):
        PesosDeEntradaALaNeurona = np.matrix(
            PesosDeEntradaALaNeurona).transpose()
        MultiplicacionMatricialDeDatosDeEntradacsvConLosPesos = np.dot(
            DatosDeEntradaCSV, PesosDeEntradaALaNeurona)
        DiferenciaCalculada = DatosDeResultadosCSV - \
            MultiplicacionMatricialDeDatosDeEntradacsvConLosPesos
        DiferenciaCalculada = np.matrix(DiferenciaCalculada).transpose()
        MultiplicacionMatricialDeDiferenciaYDatosDecsv = np.dot(
            DiferenciaCalculada, DatosDeEntradaCSV)
        PesosFinales = np.dot(
            ValorDeEta, MultiplicacionMatricialDeDiferenciaYDatosDecsv)
        PesosDeEntradaALaNeurona = np.matrix(
            PesosDeEntradaALaNeurona).transpose()
        PesosDeEntradaALaNeurona = PesosDeEntradaALaNeurona+PesosFinales
        print(PesosDeEntradaALaNeurona)
        CuadradadosDeMatrizDeLaDiferencia = np.square(DiferenciaCalculada)
        Sumatoria = np.sum(CuadradadosDeMatrizDeLaDiferencia)
        Resultado = np.sqrt(Sumatoria)
        ResultadosObtenidosPorIteracion.append(Resultado)
        Iteraciones.append(iteracion)
        print(Resultado)

    Graficar(Iteraciones, ResultadosObtenidosPorIteracion)


def TensorFlowSolve(eta, iteraciones):
    dataset = pd.read_csv('193259.csv')
    valoresDeEntrada = dataset.iloc[:, 0:3].values
    valoresDeSalida = dataset.iloc[:, 3].values
    print('valoresDeEntrada= ', len(valoresDeEntrada), "\n", valoresDeEntrada)
    print('valoresDeSalida= ', len(valoresDeSalida), "\n", valoresDeSalida)

    capa = tf.keras.layers.Dense(units=1 ,input_dim=3)
    modelo = tf.keras.Sequential([capa])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(eta),
        loss='mean_squared_error'
    )
    print('comenzando entrenamiento...')
    historial = modelo.fit(valoresDeEntrada, valoresDeSalida, epochs=iteraciones)
    print('Entrenado\n')

    resultado= modelo.predict(valoresDeEntrada)

    pyplot.xlabel("# epoca")
    pyplot.ylabel("Magnitud de perdida")
    pyplot.plot(historial.history["loss"])
    pyplot.show()

    resultado_df = pd.DataFrame(data= resultado, columns= ["Y Calculada"])
    print(resultado_df)
    resultado_df.to_csv("./output.csv")
    print("pesos")
    print(capa.get_weights())

root = Tk()
root.title('Entrenamiento de neurona')
root.geometry('700x500')

eta = Entry(root)
eta.grid(row=0, column=0)
labelX = Label(root, text="Valor de eta").grid(row=1, column=0)

iteraciones = Entry(root)
iteraciones.grid(row=0, column=1)
labelIteraciones = Label(root, text="Iteraciones").grid(row=1, column=1)


limpiar = Button(root, text="Solución con tensorflow",
                 command=lambda: TensorFlowSolve(float(eta.get()), int(iteraciones.get()))).grid(row=0, column=2)

limpiar = Button(root, text="Solución con algebra lineal",
                 command=lambda: AlgebraLinealSolve(eta.get(), iteraciones.get())).grid(row=0, column=3)

root.mainloop()
