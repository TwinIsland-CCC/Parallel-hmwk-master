#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <string>
#include <random>
#include <omp.h>
#include <barrier>

using namespace std;


void gen_arr(vector<int>& arr, int n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dis(gen);
    }
}

int reduce_add(vector<int>& arr) {
    int tmp = 0;
    for (int i = 0; i < arr.size(); i++) {
        tmp += arr[i];
    }
    return tmp;
}

int reduce_add_thread(vector<int>& arr, std::barrier<>& sync_barrier) {
    int num_threads = 8;
    vector<int> partial_sum(num_threads, 0);
    vector<std::thread> threads;
    int n = arr.size();

    for (int t = 0; t < num_threads; ++t) {
        int start = t * n / num_threads;
        int end = (t + 1) * n / num_threads;
        threads.emplace_back([&, t, start, end]() {
            int sum = 0;
            for (int i = start; i < end; ++i) {
                sum += arr[i];
            }
            partial_sum[t] = sum;
        });
    }

    for (auto& th : threads) th.join();

    int total = 0;
    for (int t = 0; t < num_threads; ++t) {
        total += partial_sum[t];
    }
    return total;
}


int main() {
    int n = 100000000;

    vector<int> arr(n);
    gen_arr(arr, n);
    vector<int> arr_copy = arr; // For baseline comparison
    auto start = chrono::high_resolution_clock::now();
    int res = reduce_add(arr);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Size: " << n << ", Baseline: " << duration << "us, res: " << res << endl;

    arr = arr_copy;

    const int max_threads = std::thread::hardware_concurrency();
    std::barrier sync_barrier(max_threads);
    start = chrono::high_resolution_clock::now();
    res = reduce_add_thread(arr, sync_barrier);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "[thread] time: " << duration << "us, res: " << res << endl;

    return 0;
}