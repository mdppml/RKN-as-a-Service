#ifndef PML_LLIB_H
#define PML_LLIB_H

#include <stdlib.h>
#include <stdint.h>

#define PARTITION_COUNT 100
#define MAX_SAMPLE_SIZE 50000

void Swap(uint64_t* a, uint64_t* b)
{
    uint64_t t = *a;
    *a = *b;
    *b = t;
}

int Partition (uint64_t real_arr[], int low, int high)
{
    uint64_t pivot = real_arr[high]; // pivot
    int i = (low - 1); // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (real_arr[j] > pivot)
        {
            i++; // increment index of smaller element
            Swap(&real_arr[i], &real_arr[j]);
        }
    }
    Swap(&real_arr[i + 1], &real_arr[high]);
    return (i + 1);
}
void QuickSort(uint64_t real_arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
         at right place */
        int pi = Partition(real_arr, low, high);

        // Separately sort elements before
        // partition and after Partition
        QuickSort(real_arr, low, pi - 1);
        QuickSort(real_arr, pi + 1, high);
    }
}

void SortValues(uint64_t real_arr[], int n){
    QuickSort(real_arr, 0, n);
}




#endif //PML_LLIB_H
