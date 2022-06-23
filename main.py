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


def Graficar(valoresX, valoresY):
    print(valoresX)
    print(valoresY)
    fig, ax = pyplot.subplots()
    ax.set_title("Grafica")
    ax.plot(valoresX, valoresY)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    pyplot.savefig("Grafica")
    pyplot.close()


def AlgebraLinealSolve(eta, iteraciones):
    datosX = []
    datosY = []
    W = [0.1, 0.3, 0.5, 0.2]
    X = pd.read_csv('193259.csv', header=None, skiprows=1, names=[
        'x1', 'x2', 'x3', 'y'], usecols=['x1', 'x2', 'x3'])
    Y = pd.read_csv('193259.csv', header=None, skiprows=1,
                    names=['x1', 'x2', 'x3', 'y'], usecols=['y'])
    N = []
    N.append(float(eta))
    X = np.matrix(X)
    X = np.c_[np.ones(100), X]
    N = np.matrix(N)
    Y = np.matrix(Y)

    for k in range(int(iteraciones)):
        W = np.matrix(W).transpose()
        U = np.dot(X, W)
        E = Y-U
        E = np.matrix(E).transpose()
        EX = np.dot(E, X)
        AW = np.dot(N, EX)
        W = np.matrix(W).transpose()
        W = W+AW
        E2 = np.square(E)
        E2Sum = np.sum(E2)
        EV = np.sqrt(E2Sum)
        datosY.append(math.ceil(EV))
        datosX.append(k)
        print(math.ceil(EV))

    Graficar(datosX, datosY)




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
