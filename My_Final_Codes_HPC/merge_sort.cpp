#include <iostream>
#include <omp.h>
#include <chrono>

// #include <bits/stdc++.h>

using namespace std;

void merge(int arr[], int start, int end)
{

    int mid = (start + end) / 2;

    // find lenght of 2 new arrays
    int len1 = mid - start + 1; // +1 as it is zero based indexing
    int len2 = end - mid;

    // create 2 new arrays
    int *first = new int[len1];
    int *second = new int[len2];

    // copy values in Both sub array
    int mainArrIdx = start;

    // copy values for 1st array
    for (int i = 0; i < len1; i++)
    {
        first[i] = arr[mainArrIdx++];
    }

    // copy values for 2nd array
    for (int i = 0; i < len2; i++)
    {
        second[i] = arr[mainArrIdx++];
    }

    // merge 2 sorted arrays
    int index1 = 0, index2 = 0;
    mainArrIdx = start; // reset mainArrIdx to start index

    while (index1 < len1 && index2 < len2)
    {
        if (first[index1] < second[index2])
        {
            arr[mainArrIdx++] = first[index1++];
        }
        else
        {
            arr[mainArrIdx++] = second[index2++];
        }
    }

    // Add Elements if remaining
    while (index1 < len1)
    {
        arr[mainArrIdx++] = first[index1++];
    }

    while (index2 < len2)
    {
        arr[mainArrIdx++] = second[index2++];
    }
}

void mergeSortSeq(int arr[], int start, int end)
{

    // Base case
    if (start >= end)
    {
        return;
    }

    int mid = (start + end) / 2;

    // Sort Left Part of Array
    mergeSortSeq(arr, start, mid);

    // Sort Right Part of Array
    mergeSortSeq(arr, mid + 1, end);

    // Merge Left & Right Array

    merge(arr, start, end);
}

void mergePara(int arr[], int start, int end)
{
    int mid = (start + end) / 2;

    int len1 = mid - start + 1;
    int len2 = end - mid;

    int *first = new int[len1];
    int *second = new int[len2];

    int mainArrIdx = start;

    for (int i = 0; i < len1; i++)
    {
        first[i] = arr[mainArrIdx++];
    }

    for (int i = 0; i < len2; i++)
    {
        second[i] = arr[mainArrIdx++];
    }

    int index1 = 0, index2 = 0;
    mainArrIdx = start;

    while (index1 < len1 && index2 < len2)
    {
        if (first[index1] < second[index2])
        {
            arr[mainArrIdx++] = first[index1++];
        }
        else
        {
            arr[mainArrIdx++] = second[index2++];
        }
    }

    while (index1 < len1)
    {
        arr[mainArrIdx++] = first[index1++];
    }

    while (index2 < len2)
    {
        arr[mainArrIdx++] = second[index2++];
    }

    delete[] first;
    delete[] second;
}

void mergeSortParallel(int arr[], int start, int end)
{
    if (start >= end)
    {
        return;
    }

    int mid = (start + end) / 2;

#pragma omp parallel sections
    {
#pragma omp section
        {
            mergeSortParallel(arr, start, mid);
        }

#pragma omp section
        {
            mergeSortParallel(arr, mid + 1, end);
        }
    }

    mergePara(arr, start, end);
}

int main()
{
    int arr[] = {5, 1, 56, 11, 99, 14, 2, 6, 0};
    int n = sizeof(arr) / sizeof(arr[0]);

    auto start_time = chrono::steady_clock ::now();

    mergeSortSeq(arr, 0, n - 1);

    auto end_time = chrono ::steady_clock ::now();

    cout << "Sorted Array = ";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    chrono::duration<double, milli> duration_seq = end_time - start_time;

    cout << "Sequential Merge Sort Execution Time: " << duration_seq.count() << " milliseconds" << endl;

    // mergeSortParallel

    start_time = chrono::steady_clock::now();

    mergeSortParallel(arr, 0, n - 1);

    end_time = chrono::steady_clock::now();

    cout << "Sorted Array: ";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    chrono::duration<double, milli> duration_para = end_time - start_time;

    cout << "Parallel Merge Sort Execution Time: " << duration_para.count() << " milliseconds" << endl;

    return 0;
}

/*

 Compile Command:

    g++ -fopenmp merge_sort.cpp -o merge_sort

 Run Command:

    ./merge_sort

 Output:

    Sorted Array = 0 1 2 5 6 11 14 56 99
    Sequential Merge Sort Execution Time: 8.621 microseconds
    Sorted Array: 0 1 2 5 6 11 14 56 99
    Parallel Merge Sort Execution Time: 371.529 microseconds



*/