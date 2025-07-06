#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>
#include <string>
#include <random>
#include <ctime>
#include <fstream>
#include <omp.h>
#include <thread>
#include <mutex>
#include <barrier>
#include <atomic>
#include <pthread.h>

#include <memory>
#include <cstdlib>

#include <immintrin.h>

#include "common.hpp"

#include "./BS_thread_pool.hpp"

using namespace std;


template<typename T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept {}
    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T* allocate(size_t n) {
        void* ptr = nullptr;
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, size_t) noexcept {
        _aligned_free(p);

    }
    template<class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

using aligned_vector = vector<int, AlignedAllocator<int, 32>>;


bool check_sorted(const vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}


bool check_sorted(const aligned_vector& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}



void bitonic_merge(vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if (dir == (arr[i] > arr[i + k])) {
                swap(arr[i], arr[i + k]);
            }
        }
        bitonic_merge(arr, low, k, dir);
        bitonic_merge(arr, low + k, k, dir);
    }
}

void bitonic_merge_omp(vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if (dir == (arr[i] > arr[i + k])) {
                int temp = arr[i];
                arr[i] = arr[i + k];
                arr[i + k] = temp;
            }
        }
        // #pragma omp task shared(arr)
        bitonic_merge(arr, low, k, dir);
        // #pragma omp task shared(arr)
        bitonic_merge(arr, low + k, k, dir);
        // #pragma omp taskwait
    }
}

void bitonic_sort(vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonic_sort(arr, low, k, true);
        bitonic_sort(arr, low + k, k, false);
        bitonic_merge(arr, low, cnt, dir);
    }
}

void padding(vector<int>& arr, int n) {
    while ((n & (n - 1)) != 0) {
        arr.push_back(numeric_limits<int>::max()); 
        n++;
    }
}

void padding_aligned(aligned_vector& arr, int n) {
    while ((n & (n - 1)) != 0) {
        arr.push_back(numeric_limits<int>::max()); 
        n++;
    }
}


void bitonic_sort(vector<int>& arr) {
    padding(arr, arr.size());
    bitonic_sort(arr, 0, arr.size(), true);
}

void bitonic_sort_task(vector<int>& arr, int low, int cnt, bool dir, int threshold) {
    if (cnt > 1) {
        int k = cnt / 2;
        if (cnt > threshold) {
            #pragma omp task shared(arr)
            bitonic_sort_task(arr, low, k, true, threshold);
            #pragma omp task shared(arr)
            bitonic_sort_task(arr, low + k, k, false, threshold);
            #pragma omp taskwait
        } else {
            bitonic_sort_task(arr, low, k, true, threshold);
            bitonic_sort_task(arr, low + k, k, false, threshold);
        }
        bitonic_merge(arr, low, cnt, dir);
    }
}

void bitonic_sort_parallel_omp(vector<int>& arr) {
    padding(arr, arr.size());
    bitonic_sort_task(arr, 0, arr.size(), true, 1024);
}

void bitonic_sort_iterative(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = 0; i < n; i++) {
                int j = i ^ stride;
                if (j > i) {
                    bool dir = ((i & size) == 0);
                    if ((arr[i] > arr[j]) == dir) {
                        swap(arr[i], arr[j]);
                    }
                }
            }
        }
    }
}

void bitonic_sort_iterative_omp(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                int j = i ^ stride;
                if (j > i) {
                    bool dir = ((i & size) == 0);
                    if ((arr[i] > arr[j]) == dir) {
                        swap(arr[i], arr[j]);
                    }
                }
            }
        }
    }
}

void bitonic_sort_iterative_thread(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();
    int num_threads = THREAD_NUM;
    BS::thread_pool pool(num_threads);
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int batch = (n + num_threads - 1) / num_threads;  // 向上取整，负载更均衡
            for (int t = 0; t < num_threads; t++) {
                int start = t * batch;
                int end = min(n, (t + 1) * batch);
                pool.detach_task([&, start, end, size, stride]() {
                    for (int i = start; i < end; i++) {
                        int j = i ^ stride;
                        if (j > i) {
                            bool dir = ((i & size) == 0);
                            if ((arr[i] > arr[j]) == dir) {
                                swap(arr[i], arr[j]);
                            }
                        }
                    }
                });
            }
            pool.wait();
        }
    }
}

struct pthread_args {
    vector<int>* arr;
    int start, end, size, stride, n;
};

void* bitonic_thread_func(void* arg) {
    pthread_args* a = (pthread_args*)arg;
    auto& arr = *(a->arr);
    for (int i = a->start; i < a->end; i++) {
        int j = i ^ a->stride;
        if (j > i) {
            bool dir = ((i & a->size) == 0);
            if ((arr[i] > arr[j]) == dir) {
                swap(arr[i], arr[j]);
            }
        }
    }
    return nullptr;
}

void bitonic_sort_iterative_pthread(vector<int>& arr, int threshold=1024) {
    padding(arr, arr.size());
    int n = arr.size();
    int num_threads = THREAD_NUM;

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (n < threshold) {
                for (int i = 0; i < n; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool dir = ((i & size) == 0);
                        if ((arr[i] > arr[j]) == dir) {
                            swap(arr[i], arr[j]);
                        }
                    }
                }
            } else {
                int batch = (n + num_threads - 1) / num_threads;
                vector<pthread_t> threads(num_threads);
                vector<pthread_args> args(num_threads);
                for (int t = 0; t < num_threads; t++) {
                    args[t] = pthread_args{&arr, t * batch, min(n, (t + 1) * batch), size, stride, n};
                    pthread_create(&threads[t], nullptr, bitonic_thread_func, &args[t]);
                }
                for (int t = 0; t < num_threads; t++) {
                    pthread_join(threads[t], nullptr);
                }
            }
        }
    }
}


void bitonic_sort_iterative_avx2(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride % 8 == 0) {
                // 一次处理8个int
                for (int i = 0; i < n; i += 2 * stride) {
                    for (int j = 0; j < stride; j += 8) {
                        if (i + j + stride + 7 < n) {
                            bool dir = ((i & size) == 0);
                            __m256i va = _mm256_loadu_si256((__m256i*)&arr[i + j]);
                            __m256i vb = _mm256_loadu_si256((__m256i*)&arr[i + j + stride]);
                            __m256i cmp = dir ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
                            __m256i va_new = _mm256_blendv_epi8(va, vb, cmp);
                            __m256i vb_new = _mm256_blendv_epi8(vb, va, cmp);
                            _mm256_storeu_si256((__m256i*)&arr[i + j], va_new);
                            _mm256_storeu_si256((__m256i*)&arr[i + j + stride], vb_new);
                        }
                    }
                }
            } else {
                // 一次处理一个
                for (int i = 0; i < n; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool dir = ((i & size) == 0);
                        if ((arr[i] > arr[j]) == dir) {
                            swap(arr[i], arr[j]);
                        }
                    }
                }
            }
        }
    }
    while (!arr.empty() && arr.back() == numeric_limits<int>::max()) arr.pop_back();
}

void bitonic_sort_iterative_avx2_omp(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();


    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride % 8 == 0) {
                // 一次处理8个int
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i += 2 * stride) {
                    for (int j = 0; j < stride; j += 8) {
                        if (i + j + stride + 7 < n) {
                            bool dir = ((i & size) == 0);
                            __m256i va = _mm256_loadu_si256((__m256i*)&arr[i + j]);
                            __m256i vb = _mm256_loadu_si256((__m256i*)&arr[i + j + stride]);
                            __m256i cmp = dir ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
                            __m256i va_new = _mm256_blendv_epi8(va, vb, cmp);
                            __m256i vb_new = _mm256_blendv_epi8(vb, va, cmp);
                            _mm256_storeu_si256((__m256i*)&arr[i + j], va_new);
                            _mm256_storeu_si256((__m256i*)&arr[i + j + stride], vb_new);
                        }
                    }
                }
            } else {
                // 一次处理一个
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool dir = ((i & size) == 0);
                        if ((arr[i] > arr[j]) == dir) {
                            swap(arr[i], arr[j]);
                        }
                    }
                }
            }
        }
    }
    while (!arr.empty() && arr.back() == numeric_limits<int>::max()) arr.pop_back();
}


void bitonic_sort_iterative_avx2_thread(vector<int>& arr) {
    padding(arr, arr.size());
    int n = arr.size();
    int num_threads = THREAD_NUM * 4;
    BS::thread_pool pool(num_threads);
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride % 8 == 0) {
                // 一次处理8个int
                int batch = (n / (2 * stride) + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; ++t) {
                    int start = t * batch * 2 * stride;
                    int end = min(n, (t + 1) * batch * 2 * stride);
                    pool.detach_task([&, start, end, size, stride]() {
                        for (int i = start; i < end; i += 2 * stride) {
                            for (int j = 0; j < stride; j += 8) {
                                if (i + j + stride + 7 < n) {
                                    bool dir = ((i & size) == 0);
                                    __m256i va = _mm256_loadu_si256((__m256i*)&arr[i + j]);
                                    __m256i vb = _mm256_loadu_si256((__m256i*)&arr[i + j + stride]);
                                    __m256i cmp = dir ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
                                    __m256i va_new = _mm256_blendv_epi8(va, vb, cmp);
                                    __m256i vb_new = _mm256_blendv_epi8(vb, va, cmp);
                                    _mm256_storeu_si256((__m256i*)&arr[i + j], va_new);
                                    _mm256_storeu_si256((__m256i*)&arr[i + j + stride], vb_new);
                                }
                            }
                        }
                    });
                }
                pool.wait();
            } else {
                // 一次处理一个
                int batch = (n + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; t++) {
                    int start = t * batch;
                    int end = min(n, (t + 1) * batch);
                    pool.detach_task([&, start, end, size, stride]() {
                        for (int i = start; i < end; i++) {
                            int j = i ^ stride;
                            if (j > i) {
                                bool dir = ((i & size) == 0);
                                if ((arr[i] > arr[j]) == dir) {
                                    swap(arr[i], arr[j]);
                                }
                            }
                        }
                    });
                }
                pool.wait();
            }
        }
    }
    while (!arr.empty() && arr.back() == numeric_limits<int>::max()) arr.pop_back();
}

void bitonic_sort_iterative_avx2_aligned(aligned_vector& arr) {
    padding_aligned(arr, arr.size());
    int n = arr.size();


    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride % 8 == 0) {
                // 一次处理8个int
                for (int i = 0; i < n; i += 2 * stride) {
                    for (int j = 0; j < stride; j += 8) {
                        if (i + j + stride + 7 < n) {
                            bool dir = ((i & size) == 0);
                            __m256i va = _mm256_load_si256((__m256i*)&arr[i + j]);
                            __m256i vb = _mm256_load_si256((__m256i*)&arr[i + j + stride]);
                            __m256i cmp = dir ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
                            __m256i va_new = _mm256_blendv_epi8(va, vb, cmp);
                            __m256i vb_new = _mm256_blendv_epi8(vb, va, cmp);
                            _mm256_store_si256((__m256i*)&arr[i + j], va_new);
                            _mm256_store_si256((__m256i*)&arr[i + j + stride], vb_new);
                        }
                    }
                }
            } else {
                // 一次处理一个
                for (int i = 0; i < n; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool dir = ((i & size) == 0);
                        if ((arr[i] > arr[j]) == dir) {
                            swap(arr[i], arr[j]);
                        }
                    }
                }
            }
        }
    }
    while (!arr.empty() && arr.back() == numeric_limits<int>::max()) arr.pop_back();
}

void gen_arr(vector<int>& arr, int n) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis;
    
    for (int i = 0; i < n; i++) {
        arr[i] = dis(gen);
    }
}

void gen_aligned_arr(aligned_vector& arr, int n) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000);
    
    for (int i = 0; i < n; i++) {
        arr[i] = dis(gen);
    }
}

void print_arr(const vector<int>& arr) {
    for (const auto& val : arr) {
        cout << val << " ";
    }
    cout << endl;
}

// int main() {
//     // vector<int> arr{10, 30, 11, 20, 4, 33, 2, 1};
//     // int n = arr.size();

//     int n = 65534;
//     cout << "Size: " << n << endl;
//     vector<int> arr(n);
//     gen_arr(arr, n);
//     vector<int> arr_copy = arr; // For baseline comparison
//     auto start = chrono::high_resolution_clock::now();
//     bitonic_sort(arr);
//     auto end = chrono::high_resolution_clock::now();
//     auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "Baseline: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;

//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_iterative(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[iterative] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;


//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     sort(arr.begin(), arr.end());
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[qsort] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;

//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_parallel_omp(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[omp] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;

//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_iterative_omp(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[iterative_omp] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;

//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_iterative_thread(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[iterative_thread] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;
    
//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_iterative_avx2(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[iterative_avx2] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;
//     // print_arr(arr);

//     arr = arr_copy;
//     start = chrono::high_resolution_clock::now();
//     bitonic_sort_iterative_avx2_omp(arr);
//     end = chrono::high_resolution_clock::now();
//     duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
//     cout << "[iterative_avx2_omp] time: " << duration << "us, sorted: " << (check_sorted(arr) ? "Y" : "N") << endl;
//     // print_arr(arr);

//     return 0;
// }