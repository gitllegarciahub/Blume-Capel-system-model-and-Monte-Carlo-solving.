import numpy as np
import matplotlib.pyplot as plt
import random
from math import exp


# -----------------------------
# PARAMETERS
# -----------------------------

N = 20                 # Lattice size (NxN)
J = 1.0                # Interaction strength
H_values = [0, 1, 3]   # External magnetic field values
TEMP_MAX = 8
temperatures = np.linspace(0.1, TEMP_MAX, TEMP_MAX * 8)

mc_steps = 1000        # Monte Carlo steps per temperature


# -----------------------------
# INITIALIZATION
# -----------------------------

def initialize_lattice(N):
    """Random ±1 spin configuration."""
    lattice = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if random.random() < 0.5:
                lattice[i, j] = -1
    return lattice


def periodic_index(i, N):
    """Apply periodic boundary conditions."""
    return i % N


# -----------------------------
# ENERGY CALCULATION
# -----------------------------

def local_energy(lattice, i, j, H):
    """Compute local energy contribution of spin (i,j)."""
    N = lattice.shape[0]

    spin = lattice[i, j]

    neighbors = (
        lattice[periodic_index(i - 1, N), j]
        + lattice[periodic_index(i + 1, N), j]
        + lattice[i, periodic_index(j - 1, N)]
        + lattice[i, periodic_index(j + 1, N)]
    )

    interaction_energy = -J * spin * neighbors
    field_energy = -H * spin

    return interaction_energy + field_energy


def total_energy(lattice, H):
    """Compute total energy of lattice."""
    N = lattice.shape[0]
    E = 0.0

    for i in range(N):
        for j in range(N):
            E += local_energy(lattice, i, j, H)

    return E / 2.0  # avoid double counting


def magnetization(lattice):
    """Compute average magnetization."""
    return np.sum(lattice) / lattice.size


# -----------------------------
# MONTE CARLO STEP
# -----------------------------

def monte_carlo_step(lattice, T, H):
    """Perform one Monte Carlo sweep."""
    N = lattice.shape[0]

    for i in range(N):
        for j in range(N):

            dE = -2 * local_energy(lattice, i, j, H)

            if dE <= 0:
                lattice[i, j] *= -1
            else:
                if random.random() < exp(-dE / T):
                    lattice[i, j] *= -1


# -----------------------------
# MAIN SIMULATION
# -----------------------------

if __name__ == "__main__":

    for H in H_values:

        avg_magnetization = []
        avg_energy = []

        for T in temperatures:

            lattice = initialize_lattice(N)

            M_values = []
            E_values = []

            for step in range(mc_steps):
                monte_carlo_step(lattice, T, H)

                M_values.append(abs(magnetization(lattice)))
                E_values.append(total_energy(lattice, H) / lattice.size)

            avg_magnetization.append(np.mean(M_values))
            avg_energy.append(np.mean(E_values))
        # Plot magnetization vs temperature
        plt.figure()
        plt.plot(temperatures, avg_magnetization, 'o')
        plt.title(f"Magnetization vs Temperature (H = {H})")
        plt.xlabel("Temperature")
        plt.ylabel("Average Magnetization")
        plt.tight_layout()
        plt.savefig(f"magnetization_H{H}.png", dpi=300)
        plt.close()
        if H==0:
            data = np.column_stack((temperatures, avg_magnetization))
            np.savetxt("ising.txt", data, header="T M")
            print("ok")
            

        # Plot energy vs temperature
        plt.figure()
        plt.plot(temperatures, avg_energy, 'o')
        plt.title(f"Energy vs Temperature (H = {H})")
        plt.xlabel("Temperature")
        plt.ylabel("Average Energy per Spin")
        plt.tight_layout()
        plt.savefig(f"energy_H{H}.png", dpi=300)
        plt.close()




