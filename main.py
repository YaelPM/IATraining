from pickletools import optimize
from tkinter import *
import math
from random import *
import pandas as pd

# TensorFlow y tf.keras
import tensorflow as tf

# Librerias de ayuda
import numpy as np
from matplotlib import pyplot


def Graficar(Iteraciones, Resultados):
    fig, grafica = pyplot.subplots()
    grafica.set_title("Grafica")
    grafica.plot(Iteraciones, Resultados)
    grafica.set_xlabel("Iteraciones")
    grafica.set_ylabel("y")
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
        'x1', 'x2', 'x3', 'y'], usecols=['x1', 'x2', 'x3'])
    DatosDeResultadosCSV = pd.read_csv('193259.csv', header=None, skiprows=1,
                                       names=['x1', 'x2', 'x3', 'y'], usecols=['y'])
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


def TensorFlowSolve():
    dataset = pd.read_csv('193259.csv')
    x = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3].values
    print('x= ', len(x), "\n", x)
    print('y= ', len(y), "\n", y)

    capa = tf.keras.layers.Dense(3, activation="sigmoid")
    modelo = tf.keras.Sequential([capa])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.0000001),
        loss='mean_squared_error'
    )
    print('comenzando entrenamiento...')
    historial = modelo.fit(x, y, epochs=100)
    print('Entrenado\n')

    # resultado= modelo.predict([-18, 10, -42])
    # print("Resultado: ", resultado)

    pyplot.xlabel("# epoca")
    pyplot.ylabel("Magnitud de perdida")
    pyplot.plot(historial.history["loss"])
    pyplot.show()


root = Tk()
root.title('Entrenamiento de neurona')
root.geometry('500x500')

eta = Entry(root)
eta.grid(row=0, column=0)
labelX = Label(root, text="Valor de eta").grid(row=1, column=0)

iteraciones = Entry(root)
iteraciones.grid(row=0, column=1)
labelIteraciones = Label(root, text="Iteraciones").grid(row=1, column=1)


limpiar = Button(root, text="AlgebraLinealSolve",
                 command=lambda: TensorFlowSolve()).grid(row=0, column=2)

# limpiar = Button(root, text="AlgebraLinealSolve",
#                  command=lambda: AlgebraLinealSolve(eta.get(), iteraciones.get())).grid(row=0, column=2)

root.mainloop()
