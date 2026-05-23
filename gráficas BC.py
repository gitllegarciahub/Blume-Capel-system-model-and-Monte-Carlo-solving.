import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

# -----------------------
# CARGAR DATOS
# -----------------------

data_m = np.loadtxt("magnetizacion.txt", skiprows=1)
data_chi = np.loadtxt("susceptibilidad.txt", skiprows=1)
data_ising=np.loadtxt("ising.txt", skiprows=1)
Jz=np.loadtxt("datos.txt",skiprows=1)
J=Jz[0]
z=Jz[1]

A_vals = np.unique(data_m[:,0])
T_vals = np.unique(data_m[:,1])

T_ising = data_ising[:, 0]
M_ising = data_ising[:, 1]

NA = len(A_vals)
NT = len(T_vals)

Matriz=np.zeros((NT,NA))
Matriz_m = np.zeros((NT, NA-1))
Chi1=np.zeros((NT,NA))
Chi = np.zeros((NT, NA-1))


Matriz = data_m[:,2].reshape(NT, NA)
iA_ising = np.argmin(A_vals)  # Δ más negativo

m_ising = Matriz[:, iA_ising]

# eliminar SOLO ese Δ del mapa
Matriz_m = np.delete(Matriz, iA_ising, axis=1)
A_vals_plot = np.delete(A_vals, iA_ising)


Chi1 = data_chi[:,2].reshape(NT, NA)

Chi = np.delete(Chi1, iA_ising, axis=1)


Chi = gaussian_filter(Chi, sigma=1)
Chi = np.maximum(Chi, 0)

# -----------------------
# PLANO MAGNETIZACIÓN
# -----------------------
plt.contourf(T_vals, A_vals_plot, Matriz_m.T, levels=50)
plt.xlabel(r"T")
plt.ylabel("Δ")
plt.title("Magnetización |m|")
plt.colorbar()
plt.savefig("magnetización")
plt.show()


# -----------------------
# LÍNEA TEÓRICA (campo medio)
# -----------------------


Delta_teo = []
T_teo = []

for T in T_vals:
    beta = 1.0 / T
    if beta * J * z > 1:
        Delta = (1/beta) * np.log(2*(beta*J*z - 1))
        Delta_teo.append(Delta)
        T_teo.append(T)

# -----------------------
# PLANO SUSCEPTIBILIDAD
# -----------------------



plt.contourf(T_vals, A_vals_plot, Chi.T, levels=50)
plt.plot(T_teo, Delta_teo,"--",label="Campo medio")
plt.xlabel(r"T")
plt.ylabel("Δ")
plt.title("Susceptibilidad χ")
plt.colorbar()
plt.legend()
plt.savefig("susceptibilidad")
plt.show()

# -----------------------
# APROXIMACIÓN MODELO DE ISING
# -----------------------



plt.plot(T_vals,m_ising,"o",label="modelo Blume-Capel")
plt.plot(T_ising,M_ising,"o",label="modelo de Ising")
plt.xlabel("T [K]")
plt.ylabel("m")
plt.title(r"Modelo Blume-Capel con $\Delta \rightarrow -\infty$")
plt.legend()
plt.savefig("comparación modelos.png")
plt.show()






# -----------------------
# m vs T para varios Δ
# -----------------------


Delta_seleccionados = [1.5, 1.7, 1.85, 2.0, 2.2]
# elegir algunos valores de Δ (índices)
indices_Delta = [np.argmin(np.abs(A_vals - d)) for d in Delta_seleccionados]

plt.figure()

for iA in indices_Delta:
    Delta_val = A_vals[iA]
    m_curve = Matriz[:, iA]   # m(T) para ese Δ
    
    plt.plot(T_vals, m_curve, label=f"Δ = {Delta_val:.2f}")

plt.xlabel("T")
plt.ylabel("|m|")
plt.title("Magnetización vs Temperatura para distintos Δ")
plt.legend()
plt.axis([0.8,3,0,1])
plt.savefig("m_vs_T_varios_Delta.png")
plt.show()



# -----------------------
# m vs Delta para varias T
# -----------------------

T_seleccionadas = [0.8, 1.0, 1.2, 1.5, 2.0]

# índices de las temperaturas elegidas
indices_T = [np.argmin(np.abs(T_vals - t)) for t in T_seleccionadas]

plt.figure()

for iT in indices_T:

    T_val = T_vals[iT]

    # magnetización para esa T variando Δ
    m_curve = Matriz_m[iT, :]


    plt.plot(A_vals_plot,
             m_curve,
             label=f"T = {T_val:.2f}")

plt.xlabel(r"$\Delta$")
plt.ylabel(r"$|m|$")
plt.title("Magnetización vs $\Delta$ para distintas temperaturas")

plt.legend()

plt.savefig("m_vs_gamma_varias_T.png")

plt.show()




# -----------------------
# DETECCIÓN DE TRANSICIONES 
# -----------------------

linea_T_2 = []
linea_A_2 = []

linea_T_1 = []
linea_A_1 = []

# umbral aproximado para separar regímenes
T_sep = 0.8

# -----------------------
# SEGUNDO ORDEN (χ máximo)
# -----------------------

for iA, Delta in enumerate(A_vals_plot):

    chi_vs_T = Chi[:, iA]

    mask_highT = T_vals > T_sep

    if np.sum(mask_highT) > 5:
        T_high = T_vals[mask_highT]
        chi_high = chi_vs_T[mask_highT]

        # suavizado ligero
        chi_high = gaussian_filter1d(chi_high, sigma=1)

        idx = np.argmax(chi_high)

        linea_T_2.append(T_high[idx])
        linea_A_2.append(Delta)


# -----------------------
# PRIMER ORDEN (barrer en Δ a T fija)
# -----------------------

linea_T_1 = []
linea_A_1 = []

T_corte = 1.0   # solo región de primer orden

for iT, T in enumerate(T_vals):

    if T > T_corte:
        continue

    m_vs_A = Matriz_m[iT, :]   # m(Δ) a T fija

    # suavizado
    m_suave = gaussian_filter1d(m_vs_A, sigma=1)

    # derivada respecto a Δ
    dm = np.abs(np.diff(m_suave))

    idx = np.argmax(dm)
    salto = dm[idx]

    # evitar ruido
    if salto < 0.05:
        continue

    Delta_trans = A_vals_plot[idx]

    linea_T_1.append(T)
    linea_A_1.append(Delta_trans)
        
# -----------------------
# PUNTO TRICRÍTICO 
# -----------------------

from scipy.spatial.distance import cdist

Delta_tc = None
T_tc = None

if len(linea_T_1) > 0 and len(linea_T_2) > 0:

    puntos_1 = np.column_stack((linea_T_1, linea_A_1))
    puntos_2 = np.column_stack((linea_T_2, linea_A_2))

    dist = cdist(puntos_1, puntos_2)

    i, j = np.unravel_index(np.argmin(dist), dist.shape)

    T_tc = (linea_T_1[i] + linea_T_2[j]) / 2
    Delta_tc = (linea_A_1[i] + linea_A_2[j]) / 2


# -----------------------
# CORTAR LÍNEA DE 2º ORDEN EN EL TRICRÍTICO
# -----------------------

if Delta_tc is not None:

    nueva_T_2 = []
    nueva_A_2 = []

    for T, A in zip(linea_T_2, linea_A_2):

        # solo parte física: Δ menor que el tricrítico
        if A <= Delta_tc:
            nueva_T_2.append(T)
            nueva_A_2.append(A)

    linea_T_2 = nueva_T_2
    linea_A_2 = nueva_A_2
# -----------------------
# PLANO CON TRANSICIONES
# -----------------------

plt.figure(figsize=(7,5))

#plano de magnetización 
plt.contourf(T_vals, A_vals_plot, Matriz_m.T, levels=50)
plt.colorbar()

# segundo orden (línea discontinua)
plt.plot(linea_T_2, linea_A_2, 'w--', linewidth=2, label='2º orden')

# primer orden (línea continua)
plt.plot(linea_T_1, linea_A_1, 'w-', linewidth=2, label='1º orden')

# punto tricrítico
if Delta_tc is not None:
    plt.plot(T_tc, Delta_tc, 'ro', markersize=8, label='Tricrítico')

plt.xlabel("T")
plt.ylabel("Δ")
plt.title("Diagrama de fases Blume-Capel")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("diagrama_tricritico.png", dpi=300)
plt.show()



import numpy as np
import matplotlib.pyplot as plt






# -----------------------------------
# Función de autocoherencia
# -----------------------------------

def f_autocoherencia(m, gamma, t):
    
    x = z * m / t
    
    return (gamma * np.sinh(x)) / (1 + gamma * np.cosh(x))

# -----------------------------------
# Valores de m
# -----------------------------------

m = np.linspace(-1.5, 1.5, 1000)

# -----------------------------------
# FIGURA
# -----------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12,5))

# ======================================================
# TRANSICIÓN DE SEGUNDO ORDEN
# ======================================================

# γ < 1/2  → segundo orden
gamma_seg = 0.3

# temperaturas:
# arriba de Tc, cerca de Tc y debajo de Tc
T_seg = [1.8, 1.2, 0.8]

for T in T_seg:
    
    f = f_autocoherencia(m, gamma_seg, T)
    
    axs[0].plot(m, f, label=f"T = {T}")

# recta y = m
axs[0].plot(m, m, 'k--', label="m")

axs[0].set_title("Transición de segundo orden")
axs[0].set_xlabel("m")
axs[0].set_ylabel("f(m)")
axs[0].legend()
axs[0].grid()

# ======================================================
# TRANSICIÓN DE PRIMER ORDEN
# ======================================================

# γ > 1/2 → región de primer orden
gamma_prim = 1.2

T_prim = [1.2, 0.9, 0.7]

for T in T_prim:
    
    f = f_autocoherencia(m, gamma_prim, T)
    
    axs[1].plot(m, f, label=f"T = {T}")

# recta y = m
axs[1].plot(m, m, 'k--', label="m")

axs[1].set_title("Transición de primer orden")
axs[1].set_xlabel("m")
axs[1].set_ylabel("f(m)")
axs[1].legend()
axs[1].grid()

# -----------------------------------
# Ajustes finales
# -----------------------------------

plt.tight_layout()

plt.savefig("autocoherencia_transiciones.png", dpi=300)

plt.show()
