/*
Title:
     Implement Min, Max, Sum and Average operations using Parallel Reduction.
*/

#include <iostream>
#include <chrono>

using namespace std;

int get_min_seq(int arr[], int n)
{
    int min_value = INT_MAX;

    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_value)
        {
            min_value = arr[i];
        }
    }

    return min_value;
}

int get_min_para(int arr[], int n)
{
    int min_value = INT_MAX;

#pragma omp paralllel for reduction(min : min_value)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_value)
        {
            min_value = arr[i];
        }
    }

    return min_value;
}

int get_max_seq(int arr[], int n)
{
    int max_value = INT_MIN;

    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
        }
    }

    return max_value;
}

int get_max_para(int arr[], int n)
{
    int max_value = INT_MIN;

#pragma omp paralllel for reduction(max : max_value)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
        }
    }

    return max_value;
}

int get_sum_seq(int arr[], int n)
{
    int sum = 0;

    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    return sum;
}

int get_sum_para(int arr[], int n)
{
    int sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    return sum;
}

double get_avg_seq(int arr[], int n)
{
    int sum = 0;

    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    double avg = sum / n;

    return avg;
}

double get_avg_para(int arr[], int n)
{
    int sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }

    double avg = sum / n;

    return avg;
}

int main()
{
    int n = 20;
    int arr[n];

    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % n;
    }

    cout << "Array: ";
    for (int i = 0; i < n; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Sequential Min Value
    auto start_time = chrono::high_resolution_clock::now();
    cout << "Minimum Value By Sequential = " << get_min_seq(arr, n) << endl;
    auto end_time = chrono::high_resolution_clock::now();

    chrono::duration<double, micro> duration_seq = end_time - start_time;

    cout << "Execution time for Finding Min Value Sequentially : " << duration_seq.count() << endl;

    // Parallel Min Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Minimum Value By Parallel = " << get_min_para(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    chrono::duration<double, micro> duration_para = end_time - start_time;

    cout << "Execution time for Finding Min Value Parallelly : " << duration_para.count() << endl;

    // Sequential max Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Maximum Value By Sequential = " << get_max_seq(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_seq = end_time - start_time;

    cout << "Execution time for Finding Max Value Sequentially : " << duration_seq.count() << endl;

    // Parallel max Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Maximum Value By Parallel = " << get_max_para(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_para = end_time - start_time;

    cout << "Execution time for Finding Max Value Parallelly : " << duration_para.count() << endl;

    // Sequential Sum Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Sum Value By Sequential = " << get_sum_seq(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_seq = end_time - start_time;

    cout << "Execution time for Finding Sum Sequentially : " << duration_seq.count() << endl;

    // Parallel Sum Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Sum Value By Parallel = " << get_sum_para(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_para = end_time - start_time;

    cout << "Execution time for Finding Sum Value Parallelly : " << duration_para.count() << endl;

    // Sequential Avg Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Avg Value By Sequential = " << get_avg_seq(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_seq = end_time - start_time;

    cout << "Execution time for Finding Avg Value Sequentially : " << duration_seq.count() << endl;

    // Parallel Avg Value
    start_time = chrono::high_resolution_clock::now();
    cout << "Avg Value By Parallel = " << get_avg_para(arr, n) << endl;
    end_time = chrono::high_resolution_clock::now();

    duration_para = end_time - start_time;

    cout << "Execution time for Finding Avg Value Parallelly : " << duration_para.count() << endl;

    return 0;
}

/*

NOTE:  Make sure you open terminal in same path where your .cpp file present

Compile Command:

    g++ -fopenmp ass_3_demo.cpp -o ass_3_demo

Run Command:

    below command for linux
    ./ass_3_demo

    below command for windows
    .\ass_3_demo

Output:

    Array: 1 7 14 0 9 4 18 18 2 4 5 5 1 7 1 11 15 2 7 16
    Minimum Value By Sequential = 0
    Execution time for Finding Min Value Sequentially : 1000.5
    Minimum Value By Parallel = 0
    Execution time for Finding Min Value Parallelly : 1000.5
    Maximum Value By Sequential = 18
    Execution time for Finding Max Value Sequentially : 1000.4
    Maximum Value By Parallel = 18
    Execution time for Finding Max Value Parallelly : 999.7
    Sum Value By Sequential = 147
    Execution time for Finding Sum Sequentially : 1001.7
    Sum Value By Parallel = 147
    Execution time for Finding Sum Value Parallelly : 0
    Avg Value By Sequential = 7
    Execution time for Finding Avg Value Sequentially : 1000.4
    Avg Value By Parallel = 7
    Execution time for Finding Avg Value Parallelly : 824.8


*/