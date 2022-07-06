from tkinter import *
import math
from random import *
import pandas as pd

# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
from matplotlib import pyplot

print(tf.__version__)


def Graficar(Iteraciones, Resultados):
    fig, grafica = pyplot.subplots()
    grafica.set_title("Grafica")
    grafica.plot(Iteraciones, Resultados)
    grafica.set_xlabel("x")
    grafica.set_ylabel("y")
    pyplot.savefig("Grafica")
    pyplot.close()


def AlgebraLinealSolve(eta, iteraciones):
    Iteraciones = []
    ResultadosObtenidosPorIteracion = []
    PesosDeEntradaALaNeurona = [0.1, 0.3, 0.5, 0.2]
    #leyendo información del archivo csv
    DatosDeEntradaCSV = pd.read_csv('193259.csv', header=None, skiprows=1, names=[
        'x1', 'x2', 'x3', 'y'], usecols=['x1', 'x2', 'x3'])
    DatosDeResultadosCSV = pd.read_csv('193259.csv', header=None, skiprows=1,
                                       names=['x1', 'x2', 'x3', 'y'], usecols=['y'])
    ValorDeEta = []
    ValorDeEta.append(float(eta))
    #Creando matrices
    DatosDeEntradaCSV = np.matrix(DatosDeEntradaCSV)
    DatosDeEntradaCSV = np.c_[np.ones(100), DatosDeEntradaCSV]
    ValorDeEta = np.matrix(ValorDeEta)
    DatosDeResultadosCSV = np.matrix(DatosDeResultadosCSV)

    for iteracion in range(int(iteraciones)):
        PesosDeEntradaALaNeurona = np.matrix(PesosDeEntradaALaNeurona).transpose()
        MultiplicacionMatricialDeDatosDeEntradacsvConLosPesos = np.dot(DatosDeEntradaCSV, PesosDeEntradaALaNeurona)
        DiferenciaCalculada = DatosDeResultadosCSV-MultiplicacionMatricialDeDatosDeEntradacsvConLosPesos
        DiferenciaCalculada = np.matrix(DiferenciaCalculada).transpose()
        MultiplicacionMatricialDeDiferenciaYDatosDecsv = np.dot(DiferenciaCalculada, DatosDeEntradaCSV)
        PesosFinales = np.dot(ValorDeEta, MultiplicacionMatricialDeDiferenciaYDatosDecsv)
        PesosDeEntradaALaNeurona = np.matrix(PesosDeEntradaALaNeurona).transpose()
        PesosDeEntradaALaNeurona = PesosDeEntradaALaNeurona+PesosFinales
        CuadradadosDeMatrizDeLaDiferencia = np.square(DiferenciaCalculada)
        Sumatoria = np.sum(CuadradadosDeMatrizDeLaDiferencia)
        Resultado = np.sqrt(Sumatoria)
        ResultadosObtenidosPorIteracion.append(math.ceil(Resultado))
        Iteraciones.append(iteracion)
        print(math.ceil(Resultado))

    Graficar(Iteraciones, ResultadosObtenidosPorIteracion)


root = Tk()
root.title('Entrenamiento de neurona')
root.geometry('500x500')

eta = Entry(root)
eta.grid(row=0, column=0)
labelX = Label(root, text="Valor de eta").grid(row=1, column=0)

iteraciones = Entry(root)
iteraciones.grid(row=0, column=1)
labelIteraciones = Label(root, text="Iteraciones").grid(row=1, column=1)


# limpiar = Button(root, text="AlgebraLinealSolve",
#                  command=lambda: AlgebraLinealSolve(eta.get(), iteraciones.get())).grid(row=0, column=2)

limpiar = Button(root, text="AlgebraLinealSolve",
                 command=lambda: AlgebraLinealSolve(eta.get(), iteraciones.get())).grid(row=0, column=2)

root.mainloop()
