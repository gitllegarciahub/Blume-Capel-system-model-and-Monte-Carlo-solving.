from math import *
import numpy as np
from random import *

N=20 #filas
M=20 #columnas
kb=1 #valor de la constante de boltzmann
S=np.array([1,0,-1])
Sp=np.zeros((N, M))  #valor del spin de cada componente
NN=40 #numero de datos de T y de A
temp=2*10**3 #numero de pasos temporales
J=1
z=4      #red cuadrada
Ain=-2
Afin=3
A = np.concatenate(([-100], np.linspace(Ain, Afin, NN)))
Tmin=0.1
Tmax=3
T = np.linspace(Tmin, Tmax, NN)
B=1/T

Matriz_m = np.zeros((NN, NN+1))   # vector para guardar magnetizacion
Chi = np.zeros((NN, NN+1))        #vector para guardar susceptibilidad
nrep = 20                        # número de repeticiones 


# -----------------------
# FUNCIÓN ΔE
# -----------------------
def deltaE(Sp, x, y, nuevo, Jn, An):
    viejo = Sp[x, y]
    
    vecinos = (
        Sp[(x+1)%N, y] +
        Sp[(x-1)%N, y] +
        Sp[x, (y+1)%M] +
        Sp[x, (y-1)%M]
    )
    
    E_old = -Jn * viejo * vecinos + An * viejo**2
    E_new = -Jn * nuevo * vecinos + An * nuevo**2
    
    return E_new - E_old


for n in range(len(A)):          # Δ
    for m in range(len(T)):      # T
        Mag_acum = 0
        Chi_acum = 0

        for rep in range(nrep):

            # inicializo los spines aleatoriamente 
            Sp = np.random.choice(S, size=(N,M))

            m_list = []

            if T[m] < 0.8:
                temp_local = 10000   # más pasos a baja T
            else:
                if T[m]<1.5:
                    temp_local= 6000
                else:
                    temp_local=3000

            for t in range(temp_local):
                for _ in range(N*M):
                    x = randint(0, N-1)
                    y = randint(0, M-1)

                    posibles = [-1,0,1]
                    posibles.remove(Sp[x,y])
                    nuevo = choice(posibles)
                        
                    dE = deltaE(Sp, x, y, nuevo, J, A[n])

                    if dE <= 0 or random() < exp(-B[m] * dE):
                        Sp[x,y] = nuevo

                # medir SOLO después de equilibrar
                if t > temp_local*0.6:
                    m_inst = np.mean(Sp)
                    m_list.append(m_inst)

            # ---- cálculo de observables ----
            m_array = np.array(m_list)

            m_mean = np.mean(m_array)
            m2_mean = np.mean(m_array**2)

            Mag_acum += np.mean(np.abs(m_array))
            Chi_acum += B[m] * N * M * (m2_mean - m_mean**2)

        # ---- promedio final ----
        Matriz_m[m, n] = Mag_acum / nrep
        Chi[m, n] = Chi_acum / nrep




#guardado de resultados

data = []

for iT in range(len(T)):
    for iA in range(len(A)):
        data.append([A[iA], T[iT], Matriz_m[iT, iA]])

data = np.array(data)

np.savetxt("magnetizacion.txt", data, header="Delta T Magnetizacion")



data2 = []

for iT in range(len(T)):
    for iA in range(len(A)):
        data2.append([A[iA], T[iT], Chi[iT, iA]])

data2 = np.array(data2)

np.savetxt("susceptibilidad.txt", data2, header="Delta T Susceptibilidad")
np.savetxt("datos.txt", [[J, z]], header="J z")
