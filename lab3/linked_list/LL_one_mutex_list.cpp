#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <omp.h>
#include <time.h>

struct list_node_s{
    int data;
    struct list_node_s* next;
};

void* test1(void*);
void* test2(void*);
int member(int value);
int insert(int value);
int Delete(int value);
void print();
void Delete_list();

struct list_node_s* head_p = NULL;
pthread_t* thread_pool;
pthread_mutex_t list_mutex;
int thread_count;

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];
    pthread_mutex_init(&list_mutex, NULL);

    for(int i=0 ; i<1000 ; ++i)
        insert(i);

    double startTime, endTime;
    srand(time(NULL));
    startTime = omp_get_wtime(); 
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, test1, NULL);

    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime(); 
    std::cout<<"Tiempo test1: "<<(endTime - startTime)<<"s\n";

    srand(time(NULL));
    startTime = omp_get_wtime(); 
    for(int t=0 ; t<thread_count ; ++t)
        pthread_create(&thread_pool[t], NULL, test2, NULL);

    for(int t=0 ; t<thread_count ; ++t)
        pthread_join(thread_pool[t], NULL);

    endTime = omp_get_wtime(); 
    std::cout<<"Tiempo test2: "<<(endTime - startTime)<<"s\n";
    
    pthread_mutex_destroy(&list_mutex);
    Delete_list();
}

void* test1(void*)
{
    int n=99900;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        pthread_mutex_lock(&list_mutex);
        member(value);
        pthread_mutex_unlock(&list_mutex);
    }

    n=50;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        pthread_mutex_lock(&list_mutex);
        insert(value);
        pthread_mutex_unlock(&list_mutex);
    }

    n=50;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        pthread_mutex_lock(&list_mutex);
        Delete(value);
        pthread_mutex_unlock(&list_mutex);
    }

    return NULL;
}

void* test2(void*)
{
    int n=80000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        pthread_mutex_lock(&list_mutex);
        member(value);
        pthread_mutex_unlock(&list_mutex);
    }

    n=10000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        pthread_mutex_lock(&list_mutex);
        insert(value);
        pthread_mutex_unlock(&list_mutex);
    }

    n=10000;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        pthread_mutex_lock(&list_mutex);
        Delete(value);
        pthread_mutex_unlock(&list_mutex);
    }

    return NULL;
}

int member(int value)
{
    struct list_node_s* curr_p = head_p;

    while(curr_p != NULL && curr_p->data < value)
        curr_p = curr_p->next;
    
    if(curr_p == NULL || curr_p->data > value)
    {
        return 0;
    }
    return 1;
}

int insert(int value)
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p == NULL || curr_p->data > value)
    {
        temp_p = new struct list_node_s;
        temp_p->data = value;
        temp_p->next = curr_p;
        if(pred_p == NULL)
            head_p = temp_p;
        else
            pred_p->next = temp_p;
        return 1;
    }
    return 0;
}

int Delete(int value)
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p != NULL && curr_p->data == value)
    {
        if(pred_p == NULL)
        {
            head_p = curr_p->next;
            delete curr_p;
        }
        else
        {
            pred_p->next = curr_p->next;
            delete curr_p;
        }
        return 1;
    }
    return 0;
}

void print()
{
    struct list_node_s* curr_p = head_p;

    while(curr_p != NULL)
    {
        std::cout<<curr_p->data<<"->";
        curr_p=curr_p->next;
    }
    std::cout<<'\n';
}

void Delete_list()
{
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;

    while(curr_p != NULL)
    {
        pred_p = curr_p;
        curr_p = curr_p->next;
        delete pred_p;
    }
    head_p = NULL;
}