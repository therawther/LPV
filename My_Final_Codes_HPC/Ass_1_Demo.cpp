/*
Title:

    Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS
*/

#include <iostream>
#include <chrono>
#include <vector>
#include <stack>
#include <queue>

using namespace std;

void bfs_sequential(vector<vector<int>> adj_list, int start_node)
{
    int noOfVertices = adj_list.size();

    queue<int> q;
    vector<bool> visited(noOfVertices, false);

    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty())
    {
        int curr_node = q.front();
        cout << curr_node << " ";

        q.pop();

        // visit neighbours
        for (int adjecent_vertex : adj_list[curr_node])
        {
            if (!visited[adjecent_vertex])
            {
                visited[adjecent_vertex] = true;
                q.push(adjecent_vertex);
            }
        }
    }
}

void bfs_parallel(vector<vector<int>> adj_list, int start_node)
{
    int noOfVertices = adj_list.size();

    queue<int> q;
    vector<bool> visited(noOfVertices, false);

    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty())
    {
        int curr_node = q.front();
        cout << curr_node << " ";

        q.pop();
// Note: omp does not support for each loop
#pragma omp parallel for
        for (int i = 0; i < adj_list[curr_node].size(); i++)
        {
            int adjecent_vertex = adj_list[curr_node][i];
            if (!visited[adjecent_vertex])
            {
                visited[adjecent_vertex] = true;
                q.push(adjecent_vertex);
            }
        }
    }
}

void dfs_sequential(vector<vector<int>> adj_list, int start_node)
{

    int no_of_vertices = adj_list.size();

    vector<int> visited(no_of_vertices, false);
    stack<int> stk;

    stk.push(start_node);
    visited[start_node] = true;

    while (!stk.empty())
    {
        int curr_node = stk.top();
        cout << curr_node << " ";

        stk.pop();

        for (int adjecent_vertex : adj_list[curr_node])
        {
            if (!visited[adjecent_vertex])
            {
                stk.push(adjecent_vertex);
                visited[adjecent_vertex] = true;
            }
        }
    }
}

void dfs_parallel(vector<vector<int>> adj_list, int start_node)
{

    int no_of_vertices = adj_list.size();

    vector<int> visited(no_of_vertices, false);
    stack<int> stk;

    stk.push(start_node);
    visited[start_node] = true;

    while (!stk.empty())
    {
        int curr_node = stk.top();
        cout << curr_node << " ";

        stk.pop();

#pragma omp parallel for
        for (int i = 0; i < adj_list[curr_node].size(); i++)
        {
            int adjecent_vertex = adj_list[curr_node][i];

            if (!visited[adjecent_vertex])
            {
                stk.push(adjecent_vertex);
                visited[adjecent_vertex] = true;
            }
        }
    }
}

int main()
{

    vector<vector<int>> adj_list = {
        {1, 4},       // neighbors of node 0
        {0, 2, 4, 3}, // neighbors of node 1
        {1, 3},       // neighbors of node 2
        {1, 2, 4},    // neighbors of node 3
        {0, 3}        // neighbors of node 4

    };

    int start_node = 0;

    // measure the execution time of BFS in sequential
    auto start_time = chrono::high_resolution_clock::now();
    cout << "BFS in sequential output: ";
    bfs_sequential(adj_list, start_node);
    auto end_time = chrono::high_resolution_clock::now();

    // auto duration_seq = chrono::duration_cast<chrono::duration<double, micro>>(end_time - start_time);
    chrono::duration<double, milli> duration_seq = end_time - start_time;

    cout << endl;
    cout << "Sequential BFS execution time: " << duration_seq.count() << " millisec" << endl;

    // measure the execution time of BFS in parallel
    start_time = chrono::high_resolution_clock::now();
    cout << "BFS in parallel output: ";
    bfs_parallel(adj_list, start_node);
    end_time = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration_para = end_time - start_time;
    cout << endl;
    cout << "Parallel BFS execution time: " << duration_para.count() << " millisec" << endl;

    // measure the execution time of DFS in sequential
    start_time = chrono::high_resolution_clock::now();
    cout << "DFS in sequential output: ";
    dfs_sequential(adj_list, start_node);
    end_time = chrono::high_resolution_clock::now();

    duration_seq = end_time - start_time;
    cout << endl;
    cout << "Sequential DFS execution time: " << duration_seq.count() << " millisec" << endl;

    cout << endl;

    // measure the execution time of DFS in parallel
    start_time = chrono::high_resolution_clock::now();
    cout << "DFS in parallel output: ";
    dfs_parallel(adj_list, start_node);
    end_time = chrono::high_resolution_clock::now();

    duration_para = end_time - start_time;

    cout << endl;

    cout << "Parallel DFS execution time: " << duration_para.count() << " millisec" << endl;

    return 0;
}

/*

NOTE:  Make sure you open terminal in same path where your .cpp file present

Compile Command:

    g++ -fopenmp Ass_1_Demo.cpp -o Ass_1_Demo

Run Command:

    for Linux
    ./Ass_1_Demo

    for Windows
    .\Ass_1_Demo

Output:

    BFS in sequential output: 0 1 4 2 3
    Sequential BFS execution time: 1465.6 microsec

    BFS in parallel output: 0 1 4 2 3
    Parallel BFS execution time: 1002.9 microsec

    DFS in sequential output: 0 4 3 2 1
    Sequential DFS execution time: 1000.9 microsec

    DFS in parallel output: 0 1 3 2 4
    Parallel DFS execution time: 999.6 microsec

Notes:

    1 second = 10^-6 seconds)
    1 millisecond = 10^-3 seconds

    <chrono> library in C++. Let's break it down:

    #include <chrono>: This is a header file that provides the functionality for time-related operations in C++. It includes classes and functions for measuring time durations, obtaining current time points, and performing time calculations.

    chrono::high_resolution_clock::now(): This is a function that returns the current time point of the high-resolution clock. It is commonly used to measure the starting and ending time of an operation.

    chrono::duration<double, micro>: This is a duration type from the <chrono> library template. It represents a duration in microseconds with a double precision floating-point value. The first template parameter specifies the type of the value, and the second template parameter specifies the ratio of the duration (in this case, microseconds).

    duration_para = end_time - start_time: This line calculates the duration of an operation by subtracting the starting time (start_time) from the ending time (end_time). The result is stored in the variable duration_para, which represents the elapsed time in microseconds.

    By using <chrono> and the provided code, you can accurately measure the execution time of specific parts of your program and perform time-related calculations.

*/