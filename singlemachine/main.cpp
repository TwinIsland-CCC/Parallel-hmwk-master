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
    int num_threads = thread::hardware_concurrency();
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
    std::vector<int>* arr;
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
                std::swap(arr[i], arr[j]);
            }
        }
    }
    return nullptr;
}

void bitonic_sort_iterative_pthread(std::vector<int>& arr, int threshold=1024) {
    padding(arr, arr.size());
    int n = arr.size();
    int num_threads = std::thread::hardware_concurrency();

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (n < threshold) {
                for (int i = 0; i < n; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool dir = ((i & size) == 0);
                        if ((arr[i] > arr[j]) == dir) {
                            std::swap(arr[i], arr[j]);
                        }
                    }
                }
            } else {
                int batch = (n + num_threads - 1) / num_threads;
                std::vector<pthread_t> threads(num_threads);
                std::vector<pthread_args> args(num_threads);
                for (int t = 0; t < num_threads; t++) {
                    args[t] = pthread_args{&arr, t * batch, std::min(n, (t + 1) * batch), size, stride, n};
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
    int num_threads = thread::hardware_concurrency();
    BS::thread_pool pool(num_threads);
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride % 8 == 0) {
                // 一次处理8个int
                int batch = (n / (2 * stride) + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; ++t) {
                    int start = t * batch * 2 * stride;
                    int end = std::min(n, (t + 1) * batch * 2 * stride);
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
                    int end = std::min(n, (t + 1) * batch);
                    pool.detach_task([&, start, end, size, stride]() {
                        for (int i = start; i < end; i++) {
                            int j = i ^ stride;
                            if (j > i) {
                                bool dir = ((i & size) == 0);
                                if ((arr[i] > arr[j]) == dir) {
                                    std::swap(arr[i], arr[j]);
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

void topknsort_qsort_recursive(int* arr, int left, int right, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
    if (left >= right) return;

    int pivot = arr[right];
    int i = left - 1;
    for (int j = left; j < right; ++j) {
        if (arr[j] <= pivot) {
            swap(arr[++i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[right]);
    int mid = i + 1;

    if (depth < max_depth) {
        // 提交两个任务到线程池
        auto future1 = pool.submit_task([=, &pool]() {
            topknsort_qsort_recursive(arr, left, mid - 1, pool, depth + 1, max_depth);
        });
        auto future2 = pool.submit_task([=, &pool]() {
            topknsort_qsort_recursive(arr, mid + 1, right, pool, depth + 1, max_depth);
        });
        // 等待两个任务完成
        future1.get();
        future2.get();
    } else {
        // 到达最大深度，转为串行排序
        sort(arr + left, arr + mid);
        sort(arr + mid + 1, arr + right + 1);
    }
}

void topknsort_qsort(vector<int>& arr) {
    int n = arr.size();
    if (n <= 1) return;
    BS::thread_pool pool(std::thread::hardware_concurrency());
    topknsort_qsort_recursive(arr.data(), 0, n - 1, ref(pool));
}


void topknsort_merge(vector<int>& arr, int left, int mid, int right, vector<int>& temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int l = left; l <= right; ++l) arr[l] = temp[l];
}


void topknsort_merge_sort(vector<int>& arr, int left, int right, vector<int>& temp, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
    if (left >= right) return;
    int mid = (left + right) / 2;

    if (depth < max_depth) {
        future<void> f1 = pool.submit_task([=, &arr, &temp, &pool] () {
            topknsort_merge_sort(ref(arr), left, mid, ref(temp), ref(pool), depth + 1, max_depth);
        });
        topknsort_merge_sort(arr, mid + 1, right, temp, pool, depth + 1, max_depth);
        f1.get();
    } else {
        sort(arr.begin() + left, arr.begin() + mid + 1);
        sort(arr.begin() + mid + 1, arr.begin() + right + 1);
    }
    topknsort_merge(arr, left, mid, right, temp);
}

void topknsort_msort(vector<int>& arr) {
    int n = arr.size();
    if (n <= 1) return;
    vector<int> temp(n);
    BS::thread_pool pool(thread::hardware_concurrency());
    topknsort_merge_sort(arr, 0, n - 1, ref(temp), ref(pool));
}

void gen_arr(vector<int>& arr, int n) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000);
    
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

int compare_int(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia > ib) - (ia < ib);  // 避免溢出
}

void test(int n, ostream& out) {
    vector<int> arr(n);
    gen_arr(arr, n);
    vector<int> arr_copy = arr;

    auto run_and_record = [&](const string& name, auto sort_func, int n) {
        long long total_time = 0;
        bool all_sorted = true;
        int repeat = 5;
        for (int r = 0; r < repeat; r++) {
            arr = arr_copy;
            cout << name << ", size: " << n << ", aligned: " << (reinterpret_cast<uintptr_t>(arr.data()) % 32 == 0) << endl;
            auto start = chrono::high_resolution_clock::now();
            sort_func(arr);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
            total_time += duration;
            if (!check_sorted(arr)) all_sorted = false;
        }
        out << n << "," << name << "," << (total_time / repeat) << "," << (all_sorted ? "Y" : "N") << "\n";
    };

    run_and_record("baseline", [](vector<int>& a){ bitonic_sort(a); }, n);
    run_and_record("iterative", [](vector<int>& a){ bitonic_sort_iterative(a); }, n);
    run_and_record("qsort", [](vector<int>& a){ sort(a.begin(), a.end()); }, n);
    run_and_record("qsort(stdlib)", [](vector<int>& a){ qsort(a.data(), a.size(), sizeof(int), compare_int); }, n);
    run_and_record("qsort_parallel", [](vector<int>& a){ topknsort_qsort(a); }, n);
    run_and_record("msort_parallel", [](vector<int>& a){ topknsort_msort(a); }, n);
    run_and_record("omp", [](vector<int>& a){ bitonic_sort_parallel_omp(a); }, n);
    run_and_record("iterative_omp", [](vector<int>& a){ bitonic_sort_iterative_omp(a); }, n);
    run_and_record("iterative_pthread", [](vector<int>& a){ bitonic_sort_iterative_pthread(a); }, n);
    run_and_record("iterative_thread", [](vector<int>& a){ bitonic_sort_iterative_thread(a); }, n);
    run_and_record("iterative_avx2", [](vector<int>& a){ bitonic_sort_iterative_avx2(a); }, n);
    run_and_record("iterative_avx2_omp", [](vector<int>& a){ bitonic_sort_iterative_avx2_omp(a); }, n);
    run_and_record("iterative_avx2_thread", [](vector<int>& a){ bitonic_sort_iterative_avx2_thread(a); }, n);
    out.flush();
    long long total_time = 0;
    bool all_sorted = true;
    int repeat = 5;
    for (int r = 0; r < repeat; r++) {
        aligned_vector aligned_arr(n);
        gen_aligned_arr(aligned_arr, n);
        cout << (reinterpret_cast<uintptr_t>(aligned_arr.data()) % 32 == 0);
        auto start = chrono::high_resolution_clock::now();
        bitonic_sort_iterative_avx2_aligned(aligned_arr);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
        check_sorted(aligned_arr);
        total_time += duration;
    }
    out << n << "," << "iterative_avx2_aligned" << "," << (total_time / repeat) << "," << (all_sorted ? "Y" : "N") << "\n";
    out.flush();
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

// 示例main函数
int main() {
    std::string filename = "benchmark_" + get_timestamp() + ".csv";
    std::ofstream fout(filename);
    fout << "data_size,method,time(us),sorted\n";
    test(2 << 12, fout);
    test(2 << 13, fout);
    test(2 << 14, fout);
    test(2 << 15, fout);
    test(2 << 16, fout);
    test(2 << 17, fout);
    test(2 << 18, fout);
    test(2 << 19, fout);
    test(2 << 20, fout);
    test(2 << 21, fout);
    fout.close();
    return 0;
}