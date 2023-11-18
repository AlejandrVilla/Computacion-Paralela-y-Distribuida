#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <iostream>
#define MAX 20000

int A[MAX],B[MAX];

void print(int* V);

int main(int argc, char* argv[])
{
    int n_threads = strtol(argv[1], NULL, 10);
    int n=MAX,i,tmp,phase;
    double sTime1, sTime2;
    double eTime1, eTime2;
    srand(time(NULL));
    for(int i=0 ; i<n ; ++i)
        A[i]=B[i]=1+rand()%n;
    // print(A);
    // print(B);

    // two parallel for directives
    sTime1=omp_get_wtime();
    for(phase=0; phase<n; ++phase)
    {
        if(phase%2==0)
        {
#           pragma omp parallel for num_threads(n_threads)\
               default(none) shared(A,n) private(i,tmp)
            for(i=1 ; i<n ; i+=2)
            {
                if(A[i-1] > A[i])
                {
                    tmp=A[i-1];
                    A[i-1]=A[i];
                    A[i]=tmp;
                }
            }
        }
        else{
#           pragma omp parallel for num_threads(n_threads)\
               default(none) shared(A,n) private(i,tmp)
            for(i=1 ; i<n-1 ; i+=2)
            {
                if(A[i] > A[i+1])
                {
                    tmp=A[i+1];
                    A[i+1]=A[i];
                    A[i]=tmp;
                }
            }            
        }
    }
    eTime1=omp_get_wtime();

    sTime2=omp_get_wtime();
    // two for directives
#   pragma omp parallel num_threads(n_threads)\
        default(none) shared(B,n) private(i,tmp,phase)
    for(phase=0; phase<n; ++phase)
    {
        if(phase%2==0)
        {
#           pragma omp for
            for(i=1; i<n; i+=2)
            {
                if(B[i-1] > B[i])
                {
                    tmp=B[i-1];
                    B[i-1]=B[i];
                    B[i]=tmp;
                }
            }
        }
        else
        {
#           pragma omp for
            for(i=1; i<n-1; i+=2)
            {
                if(B[i] > B[i+1])
                {
                    tmp=B[i+1];
                    B[i+1]=B[i];
                    B[i]=tmp;
                }
            }            
        }
    }
    eTime2=omp_get_wtime();
    std::cout<<"Two parallel for directive: "<<eTime1-sTime1<<'\n';
    std::cout<<"Two for directive: "<<eTime2-sTime2<<'\n';
    // print(A);
    // print(B);
}

void print(int* V)
{
    for(int i=0; i<MAX; ++i)
        std::cout<<V[i]<<' ';
    std::cout<<'\n';
}