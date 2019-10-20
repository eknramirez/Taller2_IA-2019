
from numpy import exp, array, random, dot
from tkinter import *
from random import choice
import matplotlib.pyplot as plt
from time import time
import numpy as np

def imprimir():
    print("Hola :)")

def esconder():
    ventana.iconify()

def Salir():
    ventana.destroy()


from tkinter import ttk
ventana = Tk()
ventana.title("Informacio General IA")
ventana.geometry("500x500")
notebook = ttk.Notebook(ventana)
notebook.pack(fill='both',expand='yes')

#Se crean las pestañas
pest0 = ttk.Frame(notebook)
pest1 = ttk.Frame(notebook)
pest2 = ttk.Frame(notebook)
pest3 = ttk.Frame(notebook)

#Se les asignan el nombre que aparece en la pestaña
notebook.add(pest0,text="Presentacion")
notebook.add(pest1,text="Perceptron")
notebook.add(pest2,text="RNS")
notebook.add(pest3,text="RNM")

#Lo que va tener en el interior de cada pestaña

#Pestaña 0 Presentacion ------------------------------------------------------------------------------
presentacion = Label(pest0,text="Presentación").pack()
estudiantes = Label(pest0,text="Estudiantes: Elkin Ramirez, Lorena Sanchez, Juan Felipe Marin").place(x=50,y=30)
perceptron = Label(pest0,text ="Perceptron: \n").place(x=50,y=60)
desper= Label(pest0,text = "Es un modelo concebido como un sistema capaz de realizar  tareas de clasificación\n de forma automática, a partir de un conjunto de ejemplo con clases diferentes. ").place(x=50,y=80)
rns = Label(pest0, text = "Redes Neuronales: ").place(x=50,y=120)
desrns = Label(pest0, text ="Paradigma de aprendizaje y procesamiento automático inspirado en el \nfuncionamiento del sistema nervioso humano.").place(x=50,y=140)

imagen1 =PhotoImage(file="pytk.gif")
larger_image =  imagen1.zoom(2, 2)
smaller_image = imagen1.subsample(2, 2)

new_image = imagen1.zoom(3, 3)
new_image = imagen1.subsample(1, 2)


ima1print = Label(pest0,image = new_image).place(x=15,y=190)

boton1 = Button(pest0,text="Saludar",fg="green",command=imprimir).place(x=400,y=400)
boton3 = Button(pest0,text="Minimizar",fg="red",command=esconder).place(x=330,y=400)
boton2 = Button(pest0,text="Cerrar",fg="red",command=Salir).place(x=450,y=400)




# Pestaña 1 Perceptron----------------------------------------------------------------------------------------

texto1 = Label(pest1, text="Perceptron").pack()

# Set de entrenamiento
setentre = Label(pest1, text="Set De Entrenamiento").pack()
texto2 = Label(pest1, text="Ingrese los parametros").place(x=60, y=50)
dat = Label(pest1, text="Dato esperado").place(x=200, y=50)

ni1 = IntVar()
ni2 = IntVar()
ni3 = IntVar()
ni4 = IntVar()
ni5 = IntVar()
ni6 = IntVar()
ni7 = IntVar()
ni8 = IntVar()
ni9 = IntVar()
ni10 = IntVar()
ni11 = IntVar()
ni12 = IntVar()

nie1 = IntVar()
nie2 = IntVar()
nie3 = IntVar()
nie4 = IntVar()

# -------------------------------------------------------------
inicia_tiempo0= time()
ingreso1 = Label(pest1, text=" 1 ").place(x=50, y=70)

n1 = Entry(pest1, textvariable=ni1).place(x=70, y=70)
n2 = Entry(pest1, textvariable=ni2).place(x=110, y=70)
n3 = Entry(pest1, textvariable=ni3).place(x=160, y=70)

ne1 = Entry(pest1, textvariable=nie1).place(x=200, y=70)

#boton_mostrar = Button(pest1, text = "Hola soy una prueba",command = lambda :print(ni1.get()))
#boton_mostrar.place(x=300,y=300)

ingreso2 = Label(pest1, text=" 2 ").place(x=50, y=100)

n4 = Entry(pest1, textvariable=ni4).place(x=70, y=100)
n5 = Entry(pest1, textvariable=ni5).place(x=110, y=100)
n6 = Entry(pest1, textvariable=ni6).place(x=160, y=100)

ne2 = Entry(pest1, textvariable=nie2).place(x=200, y=100)

ingreso3 = Label(pest1, text=" 3 ").place(x=50, y=130)

n7 = Entry(pest1, textvariable=ni7).place(x=70, y=130)
n8 = Entry(pest1, textvariable=ni8).place(x=110, y=130)
n9 = Entry(pest1, textvariable=ni9).place(x=160, y=130)

ne3 = Entry(pest1, textvariable=nie3).place(x=200, y=130)

ingreso4 = Label(pest1, text=" 4 ").place(x=50, y=160)

n10 = Entry(pest1, textvariable=ni10).place(x=70, y=160)
n11 = Entry(pest1, textvariable=ni11).place(x=110, y=160)
n12 = Entry(pest1, textvariable=ni12).place(x=160, y=160)

ne4 = Entry(pest1, textvariable=nie4).place(x=200, y=160)
# Funcion de Activación
activacion = lambda x: 0 if x < 0 else 1
ba_entrada = DoubleVar()
veces = IntVar()
ba = Label(pest1, text="Bahias: ").place(x=50, y=190)
bae = Entry(pest1, textvariable=ba_entrada).place(x=90, y=190)
inte = Label(pest1, text="# de Iteraciones:").place(x=160, y=190)
inteen = Entry(pest1, textvariable=veces).place(x=265, y=190)

bahias = ba_entrada.get()
n = veces.get()

boton_mostrar = Button(pest1, text = "Hola soy una prueba",command = lambda :print(ni1.get()))
boton_mostrar.place(x=300,y=300)

errores = []
esperados = []

# Entrenamiento
def entrena():

    w = random.rand(3)
    entrenamiento1 = [
        (array([ni1.get(), ni2.get(), ni3.get()]), nie1.get()),
        (array([ni4.get(), ni5.get(), ni6.get()]), nie2.get()),
        (array([ni7.get(), ni8.get(), ni9.get()]), nie3.get()),
        (array([ni10.get(), ni11.get(), ni12.get()]), nie4.get())
    ]
    #print(veces.get())
    for i in range(veces.get()):
        x, esperado = choice(entrenamiento1)
        resultado = dot(w, x)
        esperados.append(esperado)
        error = esperado - activacion(resultado)
        errores.append(error)
        #print("Soy prueba", resultado)
        # Ajuste

        w += ba_entrada.get() * error * x


    for x, _ in entrenamiento1:

        resultado = dot(w, x)
        print("{}: {} -> {}".format(x[:3], resultado, activacion(resultado)))
        #boton_resul = Button(pest1, text="Matriz ",command=lambda:  boton_resul.place(x=50, y=300)

        tiempo_final0 = time()
        tiempo_ejecucion0 = tiempo_final0 - inicia_tiempo0


    print("Tiempo total: ", tiempo_ejecucion0)





boton_entre = Button(pest1, text="Ejecutar", fg="green", command=entrena).place(x=230, y=220)
#ver = Label(pest1, text="Resultado: ").place(x=50, y=230)
#boton_resul =Button(pest1, text ="Resultado: ",fg= "green",command = recorrer())
#boton_resul.place(x=50,y=230)

plt.plot(errores, '-', color='red')
plt.plot(esperados, '*', color='green')



boton_1 = Button(pest1, text="Saludar", fg="green", command=imprimir).place(x=400, y=400)
boton_3 = Button(pest1, text="Minimizar", fg="red", command=esconder).place(x=330, y=400)
boton_2 = Button(pest1, text="Cerrar", fg="red", command=Salir).place(x=450, y=400)


#RNS------------------------------------------------------------------------------------------------------

inicia_tiempo1= time()
class RedNeuronal():
    def __init__(self):
        self.pesos_signaticos = 2 * random.random((3, 1)) - 1

    def __sigmoide(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoide_derivado(self, x):
        return x * (1 - x)

    def entrenamiento(self, entradas, salidas, numero_iteraciones):
        for i in range(numero_iteraciones):
            salida = self.pensar(entradas)
            error = salidas - salida
            ajuste = dot(entradas.T, error * self.__sigmoide_derivado(salida))
            self.pesos_signaticos += ajuste

    def pensar(self, entrada):
        return self.__sigmoide(dot(entrada, self.pesos_signaticos))


if __name__ == '__main__':

    r1 = IntVar()
    r2 = IntVar()
    r3 = IntVar()
    r4 = IntVar()
    r5 = IntVar()
    r6 = IntVar()
    r7 = IntVar()
    r8 = IntVar()
    r9 = IntVar()
    r10 = IntVar()
    r11 = IntVar()
    r12 = IntVar()

    rs1 = IntVar()
    rs2 = IntVar()
    rs3 = IntVar()
    rs4 = IntVar()

    rnsn = Label(pest2, text = "Red Neuronal Simples").pack()
    texto2 = Label(pest2, text="Ingrese los parametros").place(x=60, y=50)
    dat = Label(pest2, text="Dato esperado").place(x=200, y=50)

    ingre0 = Label(pest2, text=" 1 ").place(x=50, y=70)

    n1 = Entry(pest2, textvariable=r1).place(x=70, y=70)
    n2 = Entry(pest2, textvariable=r2).place(x=110, y=70)
    n3 = Entry(pest2, textvariable=r3).place(x=160, y=70)

    ne1 = Entry(pest2, textvariable=rs1).place(x=200, y=70)

    ingre1 = Label(pest2, text=" 2 ").place(x=50, y=100)

    n4 = Entry(pest2, textvariable=r4).place(x=70, y=100)
    n5 = Entry(pest2, textvariable=r5).place(x=110, y=100)
    n6 = Entry(pest2, textvariable=r6).place(x=160, y=100)

    ne2 = Entry(pest2, textvariable=rs2).place(x=200, y=100)

    ingre3 = Label(pest2, text=" 3 ").place(x=50, y=130)

    n7 = Entry(pest2, textvariable=r7).place(x=70, y=130)
    n8 = Entry(pest2, textvariable=r8).place(x=110, y=130)
    n9 = Entry(pest2, textvariable=r9).place(x=160, y=130)

    ne3 = Entry(pest2, textvariable=rs3).place(x=200, y=130)

    ingre4 = Label(pest2, text=" 4 ").place(x=50, y=160)

    n10 = Entry(pest2, textvariable=r10).place(x=70, y=160)
    n11 = Entry(pest2, textvariable=r11).place(x=110, y=160)
    n12 = Entry(pest2, textvariable=r12).place(x=160, y=160)

    ne4 = Entry(pest2, textvariable=rs4).place(x=200, y=160)
    intera = IntVar()
    ba = Label(pest2, text="Iteraciones: ").place(x=50, y=190)
    bae = Entry(pest2, textvariable=intera).place(x=120, y=190)
    red_neuronal = RedNeuronal()

    def entrenamiento1():
        entradas = array([[r1.get(), r2.get(), r3.get()], [r4.get(), r5.get(), r6.get()], [r7.get(), r8.get(), r9.get()], [r10.get(), r11.get(), r12.get()]])
        salidas = array([[rs1.get(), rs2.get(), rs3.get(), rs4.get()]]).T
        red_neuronal.entrenamiento(entradas, salidas, intera.get())
        print(red_neuronal.pesos_signaticos)
        print(red_neuronal.pensar(array([1, 0, 0])))
        tiempo_final1 = time()
        tiempo_ejecucion = tiempo_final1 - inicia_tiempo1

        print("Tiempo total: ", tiempo_ejecucion)

    boton_ejecucion =Button(pest2,text ="Ejecutar",command= entrenamiento1).place(x=230, y=220)
    #boton_resul = Button(pest2, text="Matriz ",command=lambda:print(red_neuronal.pesos_signaticos))
    #boton_resul.place(x=50, y=300)
    #boton_resulta = Button(pest2, text="Matriz ", command=lambda: print(red_neuronal.pensar(array([1, 0, 0]))))
    #boton_resul.place(x=50, y=300)
    plt.plot(red_neuronal.pesos_signaticos, '-', color='red')

boton_1 = Button(pest2, text="Saludar", fg="green", command=imprimir).place(x=400, y=400)
boton_3 = Button(pest2, text="Minimizar", fg="red", command=esconder).place(x=330, y=400)
boton_2 = Button(pest2, text="Cerrar", fg="red", command=Salir).place(x=450, y=400)

#RNM-------------------------------------------------------------------------------------------------------

presentacion = Label(pest3,text="Presentación").pack()
presentacion1 = Label(pest3,text="Red Neuronal Multiple").pack()

def sigmoide(x):
    return 1/(1 + np.exp(-x))

def sigmoide_derivado(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tangente(x):
    return np.tanh(x)

def tangente_derivada(x):
    return 1 - x**2


inicia_tiempo2 = time()

epo = IntVar()
factor = DoubleVar()

ba = Label(pest3, text="epocas: ").place(x=50, y=120)
bae = Entry(pest3, textvariable=epo).place(x=120, y=120)
ba = Label(pest3, text="Factor de aprendizaje: ").place(x=50, y=190)
bae = Entry(pest3, textvariable=factor).place(x=120, y=190)
epocas = epo.get()

class RedNeuronal():
    def __init__(self, capas, activacion='tangente'):
        if activacion == 'sigmoide':
            self.activacion = sigmoide
            self.activacion_prima = sigmoide_derivado
        elif activacion == 'tangente':
            self.activacion = tangente
            self.activacion_prima = tangente_derivada

        # Iniciarlizar pesos
        self.pesos = []
        self.deltas = []
        # capas = [2,3,2] randon entre 1, -1
        for i in range(1, len(capas) - 1):
            r = 2 * np.random.random((capas[i - 1] + 1, capas[i] + 1)) - 1
            self.pesos.append(r)

        # asignar aleatorios a la capa de salida
        r = 2 * np.random.random((capas[i] + 1, capas[i + 1])) - 1
        self.pesos.append(r)

    def ajuste(self, X, y, factor_aprendizaje=0.2, epocas=epo.get()):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epocas):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.pesos)):
                dot_value = np.dot(a[l], self.pesos[l])
                activacion = self.activacion(dot_value)
                a.append(activacion)

            # Calculo la diferencia entre la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activacion_prima(a[-1])]

            # Empezamos en la segunda capa hasta la ultima
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.pesos[l].T) * self.activacion_prima(a[l]))
            self.deltas.append(deltas)

            # invertir
            deltas.reverse()

            # Backpropagation
            for i in range(len(self.pesos)):
                capa = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.pesos[i] += factor_aprendizaje * capa.T.dot(delta)

            if k % 10000 == 0: print('epocas:', k)

    def predecir(self, x):
        unos = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.pesos)):
            a = self.activacion(np.dot(a, self.pesos[l]))
        return a

    def imprimir_pesos(self):
        print("Listado de Pesos de Conexiones")
        for i in range(len(self.pesos)):
            print(self.pesos[i])

    def obtener_deltas(self):
        return self.deltas

def hacer():
    nn = RedNeuronal([2, 3, 2], activacion='tangente')
    X = np.array([[0, 0],  # sin obstaculos
                  [0, 1],  # sin obstaculos
                  [0, -1],  # sin obstaculos
                  [0.5, 1],  # obstaculo detectado a la derecha
                  [0.5, -1],  # obstaculo a izquierdad
                  [1, 1],  # demasiado cerca a la derecha
                  [1, -1]])  # demasiado cerca a la izquierda

    y = np.array([[0, 1],  # avanzar
                  [0, 1],  # avanzar
                  [0, 1],  # avanzar
                  [-1, 1],  # giro izquierda
                  [1, 1],  # giro derecha
                  [0, -1],  # retroceder
                  [0, -1], ])  # retroceder

    nn.ajuste(X, y, factor_aprendizaje=factor.get(), epocas=epo.get())
    print(epo.get())
    index = 0
    for e in X:
        print("X: ", e, "y: ", y[index], "Red: ", nn.predecir(e))
        index = index + 1

    tiempo_final2 = time()
    tiempo_ejecucion = tiempo_final2 - inicia_tiempo2
    print("Tiempo total: ", tiempo_ejecucion)

botonr = Button(pest3, text = "Ejecutar: ", command= hacer).place(x=400,y=100)





boton_1 = Button(pest3, text="Saludar", fg="green", command=imprimir).place(x=400, y=400)
boton_3 = Button(pest3, text="Minimizar", fg="red", command=esconder).place(x=330, y=400)
boton_2 = Button(pest3, text="Cerrar", fg="red", command=Salir).place(x=450, y=400)

ventana.mainloop()