#include <iostream>
#include <mpi.h>
#define MAX 10000
using namespace std;

double A[MAX][MAX], x[MAX], y[MAX];

void ini()
{
    for (int i = 0; i < MAX; ++i)
        for (int j = 0; j < MAX; ++j)
            A[i][j] = j;
    for (int i = 0; i < MAX; ++i)
        x[i] = i, y[i] = 0;
}

void print()
{
    for (int i = 0; i < MAX; ++i)
        cout << y[i] << " ";
    cout << '\n';
}

void test()
{
    double startTime, endTime;
    ini();
    /* First pair of loops */
    startTime = MPI_Wtime();
    for (int i = 0; i < MAX; i++)
        for (int j = 0; j < MAX; j++)
            y[i] += A[i][j] * x[j];
    endTime = MPI_Wtime();
    //print();
    cout << "tiempo primer bucle: " << (endTime - startTime) << " segundos\n";
    ini();
    /* Second pair of loops */
    startTime = MPI_Wtime();
    for (int j = 0; j < MAX; j++)
        for (int i = 0; i < MAX; i++)
            y[i] += A[i][j] * x[j];
    endTime = MPI_Wtime();
    cout << "tiempo segundo bucle: " << (endTime - startTime) << " segundos\n";
    //print();
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    test();
    MPI_Finalize();
}