#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

using namespace std;

// Parámetros globales
const int N = 20;
const int M = 20;
const int NN = 40;
const int nrep = 20;

const double kb = 1.0;
const double J = 1.0;
const int z = 4;

const double Ain = -2.0;
const double Afin = 3.0;
const double Tmin = 0.1;
const double Tmax = 3.0;

// RNG
mt19937 gen(random_device{}());
uniform_real_distribution<double> dist01(0.0, 1.0);
uniform_int_distribution<int> distSpin(0, 2);

// Spins posibles
vector<int> S = {1, 0, -1};

// -----------------------
// FUNCIÓN ΔE
// -----------------------
double deltaE(vector<vector<int>>& Sp, int x, int y, int nuevo, double Jn, double An) {
    int viejo = Sp[x][y];

    int vecinos =
        Sp[(x+1)%N][y] +
        Sp[(x-1+N)%N][y] +
        Sp[x][(y+1)%M] +
        Sp[x][(y-1+M)%M];

    double E_old = -Jn * viejo * vecinos + An * viejo * viejo;
    double E_new = -Jn * nuevo * vecinos + An * nuevo * nuevo;

    return E_new - E_old;
}

int main() {

    // Crear arrays A y T
    vector<double> A;
    A.push_back(-100.0);
    for (int i = 0; i < NN; i++) {
        A.push_back(Ain + i * (Afin - Ain) / (NN - 1));
    }

    vector<double> T(NN);
    vector<double> B(NN);

    for (int i = 0; i < NN; i++) {
        T[i] = Tmin + i * (Tmax - Tmin) / (NN - 1);
        B[i] = 1.0 / T[i];
    }

    // Matrices resultados
    vector<vector<double>> Matriz_m(NN, vector<double>(NN+1, 0.0));
    vector<vector<double>> Chi(NN, vector<double>(NN+1, 0.0));

    // Distribuciones
    uniform_int_distribution<int> distX(0, N-1);
    uniform_int_distribution<int> distY(0, M-1);

    for (int n = 0; n < A.size(); n++) {

        for (int m = 0; m < T.size(); m++) {

            // -----------------------
            // PASOS MONTE CARLO DEPENDIENTES DE T
            // -----------------------
            int temp_local;

            if (T[m] < 0.8) {
                temp_local = 10000;   // muy baja T
            } else if (T[m] < 1.5) {
                temp_local = 6000;    // zona crítica
            } else {
                temp_local = 3000;    // alta T
            }

            double Mag_acum = 0.0;
            double Chi_acum = 0.0;

            for (int rep = 0; rep < nrep; rep++) {

                // Inicializar spines aleatorios
                vector<vector<int>> Sp(N, vector<int>(M));
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < M; j++)
                        Sp[i][j] = S[distSpin(gen)];

                vector<double> m_list;

                for (int t = 0; t < temp_local; t++) {

                    for (int k = 0; k < N*M; k++) {
                        int x = distX(gen);
                        int y = distY(gen);

                        vector<int> posibles = {-1, 0, 1};
                        posibles.erase(remove(posibles.begin(), posibles.end(), Sp[x][y]), posibles.end());

                        int nuevo = posibles[rand() % posibles.size()];

                        double dE = deltaE(Sp, x, y, nuevo, J, A[n]);

                        if (dE <= 0 || dist01(gen) < exp(-B[m] * dE)) {
                            Sp[x][y] = nuevo;
                        }
                    }

                    // medir tras termalización
                    if (t > temp_local * 0.6) {
                        double suma = 0.0;
                        for (int i = 0; i < N; i++)
                            for (int j = 0; j < M; j++)
                                suma += Sp[i][j];

                        double m_inst = suma / (N * M);
                        m_list.push_back(m_inst);
                    }
                }

                // ---- observables ----
                double m_mean = 0.0, m2_mean = 0.0;

                for (double val : m_list) {
                    m_mean += val;
                    m2_mean += val * val;
                }

                m_mean /= m_list.size();
                m2_mean /= m_list.size();

                double abs_mean = 0.0;
                for (double val : m_list)
                    abs_mean += abs(val);

                abs_mean /= m_list.size();

                Mag_acum += abs_mean;
                Chi_acum += B[m] * N * M * (m2_mean - m_mean * m_mean);
            }

            Matriz_m[m][n] = Mag_acum / nrep;
            Chi[m][n] = Chi_acum / nrep;
        }
    }

    // -----------------------
    // GUARDAR RESULTADOS
    // -----------------------

    ofstream f1("magnetizacion.txt");
    f1 << "Delta T Magnetizacion\n";

    for (int iT = 0; iT < T.size(); iT++) {
        for (int iA = 0; iA < A.size(); iA++) {
            f1 << A[iA] << " " << T[iT] << " " << Matriz_m[iT][iA] << "\n";
        }
    }
    f1.close();

    ofstream f2("susceptibilidad.txt");
    f2 << "Delta T Susceptibilidad\n";

    for (int iT = 0; iT < T.size(); iT++) {
        for (int iA = 0; iA < A.size(); iA++) {
            f2 << A[iA] << " " << T[iT] << " " << Chi[iT][iA] << "\n";
        }
    }
    f2.close();

    ofstream f3("datos.txt");
    f3 << "J z\n";
    f3 << J << " " << z << endl;
    f3.close();

    return 0;
}