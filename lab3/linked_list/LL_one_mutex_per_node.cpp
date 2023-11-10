#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <omp.h>
#include <time.h>

struct list_node_s{
    int data;
    struct list_node_s* next;
    pthread_mutex_t mutex;
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
int thread_count;

int main(int argc, char* argv[])
{
    thread_count = strtol(argv[1],NULL,10);
    thread_pool = new pthread_t[thread_count*sizeof(pthread_t)];

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

    Delete_list();
}

void* test1(void*)
{
    int n=99900;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        member(value);
    }

    n=50;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        insert(value);
    }

    n=50;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        Delete(value);
    }
    return NULL;
}

void* test2(void*)
{
    int n=100000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        member(value);
    }

    n=10000;
    while(n--)
    {
        int value=10000 + rand() % (20000 - 10000 + 1);
        insert(value);
    }

    n=10000;
    while(n--)
    {
        int value=100000 + rand() % (200000 - 100000 + 1);
        Delete(value);
    }
    return NULL;
}

int member(int value)
{
    struct list_node_s* temp_p = NULL;

    pthread_mutex_lock(&(head_p->mutex));
    temp_p = head_p;

    while(temp_p != NULL && temp_p->data < value)
    {
        if(temp_p->next != NULL)
            pthread_mutex_lock(&(temp_p->next->mutex));
        pthread_mutex_unlock(&(temp_p->mutex));
        temp_p = temp_p->next;
    }
    
    if(temp_p == NULL || temp_p->data > value)
    {
        if(temp_p != NULL)
            pthread_mutex_unlock(&(temp_p->mutex));
        return 0;
    }
    pthread_mutex_unlock(&(temp_p->mutex));
    return 1;
}

int insert(int value)
{
    if(head_p != NULL)
        pthread_mutex_lock(&(head_p->mutex));

    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;
    struct list_node_s* temp_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        if(curr_p->next != NULL)
            pthread_mutex_lock(&(curr_p->next->mutex));
        if(pred_p != NULL)
            pthread_mutex_unlock(&(pred_p->mutex));
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p == NULL || curr_p->data > value)
    {
        temp_p = new struct list_node_s;
        temp_p->data = value;
        temp_p->next = curr_p;
        pthread_mutex_init(&(temp_p->mutex), NULL);
        if(pred_p == NULL)
            head_p = temp_p;
        else
            pred_p->next = temp_p;
        if(curr_p != NULL)
            pthread_mutex_unlock(&(curr_p->mutex));
        if(pred_p != NULL)
            pthread_mutex_unlock(&(pred_p->mutex));
        return 1;
    }
    if(curr_p != NULL)
        pthread_mutex_unlock(&(curr_p->mutex));
    if(pred_p != NULL)
        pthread_mutex_unlock(&(pred_p->mutex));
    return 0;
}

int Delete(int value)
{
    if(head_p != NULL)
        pthread_mutex_lock(&(head_p->mutex));
    struct list_node_s* curr_p = head_p;
    struct list_node_s* pred_p = NULL;

    while(curr_p != NULL && curr_p->data < value)
    {
        if(curr_p->next != NULL)
            pthread_mutex_lock(&(curr_p->next->mutex));
        if(pred_p != NULL)
            pthread_mutex_unlock(&(pred_p->mutex));
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if(curr_p != NULL && curr_p->data == value)
    {
        if(pred_p == NULL)
        {
            head_p = curr_p->next;
            pthread_mutex_unlock(&(curr_p->mutex));
            pthread_mutex_destroy(&(curr_p->mutex));
            delete curr_p;
        }
        else
        {
            pred_p->next = curr_p->next;
            pthread_mutex_unlock(&(curr_p->mutex));
            pthread_mutex_destroy(&(curr_p->mutex));
            delete curr_p;
        }
        if(pred_p != NULL)
            pthread_mutex_unlock(&(pred_p->mutex));
        return 1;
    }
    if(curr_p != NULL)
        pthread_mutex_unlock(&(curr_p->mutex));
    if(pred_p != NULL)
        pthread_mutex_unlock(&(pred_p->mutex));
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
        pthread_mutex_destroy(&(curr_p->mutex));
        pred_p = curr_p;
        curr_p = curr_p->next;
        delete pred_p;
    }
    head_p = NULL;
}