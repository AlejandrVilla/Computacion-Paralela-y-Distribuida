#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#define vi vector<int>
#define vvi vector<vi>
using namespace std;

vvi MA, MB, MC, MC2;
int n, m, p;

void mult_matrix(vvi& R, vvi& A, vvi& B)
{
    for (int f = 0; f < n; ++f)
        for (int c = 0; c < p; ++c)
            for (int k = 0; k < m; ++k)
                R[f][c] += A[f][k] * B[k][c];
}

void mult_matrix_2(vvi& R, vvi& A, vvi& B)
{
    int blockSize = 32;
    for (int i = 0; i < n; i += blockSize)
        for (int j = 0; j < p; j += blockSize)
            for (int k = 0; k < m; k += blockSize)
                // Multiplicación de bloques
                for (int ii = i; ii < min(i + blockSize, n); ii++)
                    for (int jj = j; jj < min(j + blockSize, p); jj++)
                        for (int kk = k; kk < min(k + blockSize, m); kk++)
                            R[ii][jj] += A[ii][kk] * B[kk][jj];
}

void ini(vvi& M)
{
    for (int i = 0; i < M.size(); ++i)
        for (int j = 0; j < M[i].size(); ++j)
            M[i][j] = i + j;
}

void print(vvi& M)
{
    for (int i = 0; i < M.size(); ++i)
    {
        for (int j = 0; j < M[i].size(); ++j)
            cout << M[i][j] << ' ';
        cout << '\n';
    }
}

bool eq_Matrix(vvi& M1, vvi& M2)
{
    for (int i = 0; i < M1.size(); ++i)
        for (int j = 0; j < M1[i].size(); ++j)
            if (M1[i][j] != M2[i][j])
                return false;
    return true;
}

int main(int argc, char** argv)
{
    n = 100, m = 100, p = 10;
    MA.assign(n, vi(m, 0));
    MB.assign(m, vi(p, 0));
    MC.assign(n, vi(p, 0));
    MC2.assign(n, vi(p, 0));
    ini(MA);
    ini(MB);

    double startTime, endTime;
    MPI_Init(&argc, &argv);

    startTime = MPI_Wtime();
    mult_matrix(MC, MA, MB);
    endTime = MPI_Wtime();
    cout << "multiplicacion estandar: " << endTime - startTime << " segundos\n";

    startTime = MPI_Wtime();
    mult_matrix_2(MC2, MA, MB);
    endTime = MPI_Wtime();
    cout << "multiplicacion por bloques: " << endTime - startTime << " segundos\n";

    MPI_Finalize();

    if (eq_Matrix(MC, MC2))
        cout << "iguales\n";
    else
        cout << "no iguales\n";

    return 0;
}