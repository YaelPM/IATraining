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

limpiar = Button(root, text="AlgebraLinealSolve").grid(row=0, column=2)

root.mainloop()
