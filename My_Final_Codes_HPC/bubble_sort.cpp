#include <iostream>
#include <chrono>

using namespace std;

void swapSeq(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

void bubbleSortSeq(int arr[], int n)
{
    // require n-1 rounds
    for (int i = 0; i < n - 1; i++) // Loop for pass, require n-1 rounds
    {
        bool isSwapped = false;
        // here for n element n-1 comparision
        for (int j = 0; j < n - i - 1; j++) // loop for comparision,
        {
            if (arr[j] > arr[j + 1])
            {
                swapSeq(arr[j], arr[j + 1]);
                isSwapped = true;
            }
        }
        if (isSwapped == false)
        {
            // Array is already Sorted
            break;
        }
    }
}

void swapPara(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

void bubbleSortParallel(int arr[], int n)
{
    // require n-1 rounds
    for (int i = 0; i < n - 1; i++) // Loop for pass, require n-1 rounds
    {
        bool isSwapped = false;

#pragma omp parallel for shared(arr, isSwapped)
        for (int j = 0; j < n - i - 1; j++) // loop for  n-1 comparision,
        {
            if (arr[j] > arr[j + 1])
            {
                swapPara(arr[j], arr[j + 1]);
                isSwapped = true;
            }
        }

        if (isSwapped == false)
        {
            // Array is already Sorted
            break;
        }
    }
}

void Display(int arr[], int n)
{
    cout << "Array = ";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

    cout << endl;
}

int main()
{
    int arr[] = {4, 3, 7, 1, 5, 2, 11, 9};
    int n = sizeof(arr) / sizeof(arr[0]);

    auto start_time = chrono::steady_clock::now();
    bubbleSortSeq(arr, n);
    auto end_time = chrono::steady_clock::now();

    chrono::duration<double, micro> duration_seq = end_time - start_time;

    Display(arr, n);

    cout << "Sequential Bubble Sort Execution Time: " << duration_seq.count() << " microseconds" << endl;

    start_time = chrono::steady_clock::now();
    bubbleSortParallel(arr, n);
    end_time = chrono::steady_clock::now();

    chrono::duration<double, micro> duration_para = end_time - start_time;

    Display(arr, n);

    cout << "Parallel Bubble Sort Execution Time: " << duration_para.count() << " microseconds" << endl;

    return 0;
}

/*

Compile Command:

    g++ -fopenmp bubble_sort.cpp -o bubble_sort

Run Command:

    .\bubble_sort

Output:

    Array = 1 2 3 4 5 7 9 11
    Sequential Bubble Sort Execution Time: 0.41 microseconds
    Array = 1 2 3 4 5 7 9 11
    Parallel Bubble Sort Execution Time: 330.887 microseconds

Notes:

     Bubble sort is not an ideal algorithm for parallelization due to its inherent sequential nature and high number of data dependencies. Parallelizing the inner loop using OpenMP may not provide significant performance improvements and can even lead to additional overhead.
*/