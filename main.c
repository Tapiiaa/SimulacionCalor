#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Inicializa la matriz con 100 en los bordes y 0 en el interior
void inicializar(float* matriz, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            matriz[i * N + j] = (i == 0 || i == N-1 || j == 0 || j == N-1) ? 100.0 : 0.0;
}

// Devuelve el máximo cambio entre dos matrices
float calcular_max_cambio(float* a, float* b, int N) {
    float max = 0.0;
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j) {
            float diff = fabs(a[i * N + j] - b[i * N + j]);
            if (diff > max) max = diff;
        }
    return max;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s N umbral max_iteraciones\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    float umbral = atof(argv[2]);
    int max_iter = atoi(argv[3]);

    float* actual = malloc(N * N * sizeof(float));
    float* nuevo  = malloc(N * N * sizeof(float));
    if (!actual || !nuevo) {
        fprintf(stderr, "Error al asignar memoria\n");
        return 1;
    }

    inicializar(actual, N);
    inicializar(nuevo, N);

    double inicio = omp_get_wtime();

    int iter = 0;
    float cambio_max;

    do {
        // Paralelización del cálculo de temperaturas
#pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                nuevo[i * N + j] = 0.25 * (actual[(i-1) * N + j] +
                                          actual[(i+1) * N + j] +
                                          actual[i * N + (j-1)] +
                                          actual[i * N + (j+1)]);
            }
        }

        // Calcular el cambio máximo con reducción paralela
        cambio_max = 0.0f;
#pragma omp parallel for reduction(max:cambio_max) collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                float diff = fabs(actual[i * N + j] - nuevo[i * N + j]);
                if (diff > cambio_max) cambio_max = diff;
            }
        }

        // Intercambiar punteros
        float* temp = actual;
        actual = nuevo;
        nuevo = temp;

        iter++;
    } while (cambio_max >= umbral && iter < max_iter);

    double fin = omp_get_wtime();
    printf("Iteraciones: %d\n", iter);
    printf("Tiempo: %.4f segundos\n", fin - inicio);

    free(actual);
    free(nuevo);
    return 0;
}
