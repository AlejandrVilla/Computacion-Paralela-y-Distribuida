#include <iostream>
#include <stdlib.h>
#include <stdio.h>

struct list_node_s{
    int data;
    struct list_node_s* next;
};

int member(int value, struct list_node_s* head_p);
int insert(int value, struct list_node_s** head_p );
int Delete(int value, struct list_node_s** head_p);
void print(struct list_node_s* head_p);
void Delete_list(struct list_node_s* head_p);

int main(int argc, char* argv[])
{
    struct list_node_s* head_p = NULL;
    insert(5, &head_p);
    insert(8, &head_p);
    insert(3, &head_p);
    insert(10, &head_p);
    insert(1, &head_p);

    print(head_p);

    Delete(1, &head_p);
    Delete(10, &head_p);
    Delete(5, &head_p);
    print(head_p);

    if(member(10, head_p))
        std::cout<<"si esta\n";
    else
        std::cout<<"no esta\n";

    Delete_list(head_p);
}

int member(int value, struct list_node_s* head_p)
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

int insert(int value, struct list_node_s** head_p )
{
    struct list_node_s* curr_p = *head_p;
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
            *head_p = temp_p;
        else
            pred_p->next = temp_p;
        return 1;
    }
    return 0;
}

int Delete(int value, struct list_node_s** head_p)
{
    struct list_node_s* curr_p = *head_p;
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
            *head_p = curr_p->next;
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

void print(struct list_node_s* head_p)
{
    struct list_node_s* curr_p = head_p;

    while(curr_p != NULL)
    {
        std::cout<<curr_p->data<<"->";
        curr_p=curr_p->next;
    }
    std::cout<<'\n';
}

void Delete_list(struct list_node_s* head_p)
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