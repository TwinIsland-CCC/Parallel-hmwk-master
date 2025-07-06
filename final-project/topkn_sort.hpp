#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <immintrin.h>  

#include "BS_thread_pool.hpp"

#include "common.hpp"

using namespace std;

inline void swap_simple(int& a, int& b) {
  int temp = a;
  a = b;
  b = temp;
}

void qsort_simple(int* arr, int left, int right) {
  if (left >= right) return;

  const int pivot = arr[right];
  int i = left - 1;
  for (int j = left; j < right; ++j) {
    if (arr[j] <= pivot) {
      ++i;
      swap_simple(arr[i], arr[j]);
    }
  }
  swap_simple(arr[i + 1], arr[right]);
  qsort_simple(arr, left, i);
  qsort_simple(arr, i + 2, right);
}

void qsort_avx256(int* arr, int left, int right) {
  if (left >= right) return;
  int n = right - left + 1;
  if (n < 64) {
    qsort_simple(arr, left, right);
    return;
  }

  const int pivot = arr[right];
  int i = left - 1;
  int j = left;
  __m256i pivot_avx256 = _mm256_set1_epi32(pivot);
  int* buf = new int[n];
  int l = 0, r = n - 1;

  for (; j + 8 <= right; j += 8) {
    __m256i va = _mm256_loadu_si256((__m256i*)(arr + j));
    __m256i cmp = _mm256_cmpgt_epi32(pivot_avx256, va); // pivot > arr[j]  → arr[j] <= pivot
    alignas(32) int buf_cmp[8];
    _mm256_store_si256((__m256i*)buf_cmp, cmp);
    for (int k = 0; k < 8; ++k) {
      if (buf_cmp[k] != 0) {
        ++i;
        swap_simple(arr[i], arr[j + k]);
      }
    }
  }

  for (; j < right; ++j) {
    if (arr[j] <= pivot) {
      ++i;
      swap_simple(arr[i], arr[j]);
    }
  }

  swap_simple(arr[i + 1], arr[right]);
  qsort_avx256(arr, left, i);
  qsort_avx256(arr, i + 2, right);
}

void topknsort_qsort_recursive(int* arr, int left, int right, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
  if (left >= right) return;

  int pivot = arr[right];
  int i = left - 1;
  for (int j = left; j < right; ++j) {
    if (arr[j] <= pivot) {
      ++i;
      swap(arr[i], arr[j]);
    }
  }
  swap(arr[i + 1], arr[right]);
  int mid = i + 1;

  if (depth < max_depth) {
    auto future = pool.submit_task([=, &pool]() {
      topknsort_qsort_recursive(arr, left, mid - 1, pool, depth + 1, max_depth);
    });
    topknsort_qsort_recursive(arr, mid + 1, right, pool, depth + 1, max_depth);
    future.get();
  } else {
    sort(arr + left, arr + mid);
    sort(arr + mid + 1, arr + right + 1);
  }
}

void topknsort_qsort(vector<int>& arr) {
  int n = arr.size();
  if (n <= 1) return;
  BS::thread_pool pool(THREAD_NUM);
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

void topknsort_merge_avx256(std::vector<int>& arr, int left, int mid, int right, std::vector<int>& temp) {
  int i = left, j = mid + 1, k = left;
  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) temp[k++] = arr[i++];
    else temp[k++] = arr[j++];
  }
  // 简单的SIMD作用有限，这里用于加速剩余部分的拷贝
  while (i + 8 <= mid + 1) {
    __m256i va = _mm256_loadu_si256((__m256i*)&arr[i]);
    _mm256_storeu_si256((__m256i*)&temp[k], va);
    i += 8;
    k += 8;
  }
  while (j + 8 <= right + 1) {
    __m256i vb = _mm256_loadu_si256((__m256i*)&arr[j]);
    _mm256_storeu_si256((__m256i*)&temp[k], vb);
    j += 8;
    k += 8;
  }
  // 别漏了
  while (i <= mid) temp[k++] = arr[i++];
  while (j <= right) temp[k++] = arr[j++];
  for (int l = left; l <= right; ++l) arr[l] = temp[l];
}

void topknsort_merge_sort_avx256(vector<int>& arr, int left, int right, vector<int>& temp, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
  if (left >= right) return;
  int mid = (left + right) / 2;

  if (depth < max_depth) {
    future<void> future = pool.submit_task([=, &arr, &temp, &pool] () {
      topknsort_merge_sort_avx256(ref(arr), left, mid, ref(temp), ref(pool), depth + 1, max_depth);
    });
    topknsort_merge_sort_avx256(arr, mid + 1, right, temp, pool, depth + 1, max_depth);
    future.get();
  } else {
    sort(arr.begin() + left, arr.begin() + mid + 1);
    sort(arr.begin() + mid + 1, arr.begin() + right + 1);
  }
  topknsort_merge_avx256(arr, left, mid, right, temp);
}

void topknsort_merge_sort(vector<int>& arr, int left, int right, vector<int>& temp, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
  if (left >= right) return;
  int mid = (left + right) / 2;

  if (depth < max_depth) {
    future<void> future = pool.submit_task([=, &arr, &temp, &pool] () {
      topknsort_merge_sort(ref(arr), left, mid, ref(temp), ref(pool), depth + 1, max_depth);
    });
    topknsort_merge_sort(arr, mid + 1, right, temp, pool, depth + 1, max_depth);
    future.get();
  } else {
    sort(arr.begin() + left, arr.begin() + mid + 1);
    sort(arr.begin() + mid + 1, arr.begin() + right + 1);
  }
  topknsort_merge(arr, left, mid, right, temp);
}

void topknsort_merge_sort_thread_avx256(vector<int>& arr, int left, int right, vector<int>& temp, BS::thread_pool<>& pool, int depth = 0, int max_depth = 4) {
  if (left >= right) return;
  int mid = (left + right) / 2;

  if (depth < max_depth) {
    future<void> future = pool.submit_task([=, &arr, &temp, &pool] () {
      topknsort_merge_sort_thread_avx256(ref(arr), left, mid, ref(temp), ref(pool), depth + 1, max_depth);
    });
    topknsort_merge_sort_thread_avx256(arr, mid + 1, right, temp, pool, depth + 1, max_depth);
    future.get();
  } else {
    sort(arr.begin() + left, arr.begin() + mid + 1);
    sort(arr.begin() + mid + 1, arr.begin() + right + 1);
  }
  topknsort_merge_avx256(arr, left, mid, right, temp);
}

void topknsort_merge_sort_simple(vector<int>& arr, int left, int right, vector<int>& temp) {
  if (left >= right) return;
  int mid = (left + right) / 2;
  topknsort_merge_sort_simple(arr, left, mid, temp);
  topknsort_merge_sort_simple(arr, mid + 1, right, temp);
  topknsort_merge(arr, left, mid, right, temp);
}

void topknsort_msort_simple(vector<int>& arr) {
  int n = arr.size();
  if (n <= 1) return;
  vector<int> temp(n);
  topknsort_merge_sort_simple(arr, 0, n - 1, ref(temp));
}

void topknsort_msort(vector<int>& arr) {
  int n = arr.size();
  if (n <= 1) return;
  vector<int> temp(n);
  BS::thread_pool pool(THREAD_NUM);
  topknsort_merge_sort(arr, 0, n - 1, ref(temp), ref(pool));
}

void topknsort_msort_avx256(vector<int>& arr) {
  int n = arr.size();
  if (n <= 1) return;
  vector<int> temp(n);
  BS::thread_pool pool(THREAD_NUM);
  topknsort_merge_sort_avx256(arr, 0, n - 1, ref(temp), ref(pool));
}

void topknsort_msort_thread_avx256(vector<int>& arr) {
  int n = arr.size();
  if (n <= 1) return;
  vector<int> temp(n);
  BS::thread_pool pool(THREAD_NUM);
  topknsort_merge_sort_thread_avx256(arr, 0, n - 1, ref(temp), ref(pool));
}

