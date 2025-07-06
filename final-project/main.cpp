#include "arg_min_max_fusion.h"
#include "bitonic_sort.hpp"
#include "topk1.hpp"
#include "topkn_sort.hpp"

void topk1int32simple_cuda(const std::vector<int>& input, int& output, int& output_value, bool get_max);

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
      auto start = chrono::high_resolution_clock::now();
      sort_func(arr);
      auto end = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
      total_time += duration;
      cout << name << ", size: " << n << ", aligned: " << (reinterpret_cast<uintptr_t>(arr.data()) % 32 == 0) << "time(us):" << duration << endl;
      if (!check_sorted(arr)) all_sorted = false;
    }
    out << n << "," << name << "," << (total_time / repeat) << "," << (all_sorted ? "Y" : "N") << "\n";
  };

  run_and_record("baseline", [](vector<int>& a){ bitonic_sort(a); }, n);
  run_and_record("iterative", [](vector<int>& a){ bitonic_sort_iterative(a); }, n);
  run_and_record("qsort", [](vector<int>& a){ sort(a.begin(), a.end()); }, n);
  run_and_record("qsort(stdlib)", [](vector<int>& a){ qsort(a.data(), a.size(), sizeof(int), compare_int); }, n);
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


void test_topk1(int n, ostream& out) {
  vector<int> arr(n);
  gen_arr(arr, n);

  auto run_and_record = [&](const string& name, auto topk1_func, int n) {
    long long total_time = 0;
    int repeat = 5;
    for (int r = 0; r < repeat; r++) {
      int topk1 = 0, topk1_value = 0;
      auto start = chrono::high_resolution_clock::now();
      topk1_func(arr, topk1, topk1_value, true);
      auto end = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
      total_time += duration;
      cout << name << ", size: " << n << ", topk1=" << topk1 << ", topk1_value=" << topk1_value << ", time(us):" << duration << endl;
    }
    for (int r = 0; r < repeat; r++) {
      int topk1 = 0, topk1_value = 0;
      auto start = chrono::high_resolution_clock::now();
      topk1_func(arr, topk1, topk1_value, false);
      auto end = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
      total_time += duration;
      cout << name << ", size: " << n << ", topk1=" << topk1 << ", topk1_value=" << topk1_value << ", time(us):" << duration << endl;
    }
    out << n << "," << name << "," << (total_time / (repeat * 2)) << "," << "\n";
  };

  run_and_record("baseline", [](vector<int>& a, int& output, int& output_value, bool get_max){ topk1int32simple(a, output, output_value, get_max); }, n);
  run_and_record("avx256", [](vector<int>& a, int& output, int& output_value, bool get_max){ topk1int32simple_avx256(a, output, output_value, get_max); }, n);
  run_and_record("thread_pool", [](vector<int>& a, int& output, int& output_value, bool get_max){ topk1int32simple_thread(a, output, output_value, get_max); }, n);
  run_and_record("thread_pool+avx256", [](vector<int>& a, int& output, int& output_value, bool get_max){ topk1int32simple_thread_avx256(a, output, output_value, get_max); }, n);
  out.flush();
}

void test_topkn(int n, ostream& out) {
  vector<int> arr(n);
  gen_arr(arr, n);
  vector<int> arr_copy = arr;

  auto run_and_record = [&](const string& name, auto sort_func, int n) {
    long long total_time = 0;
    bool all_sorted = true;
    int repeat = 5;
    for (int r = 0; r < repeat; r++) {
      arr = arr_copy;
      auto start = chrono::high_resolution_clock::now();
      sort_func(arr);
      auto end = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
      total_time += duration;
      cout << name << ", size: " << n << ", aligned: " << (reinterpret_cast<uintptr_t>(arr.data()) % 32 == 0) << "time(us):" << duration << endl;
      if (!check_sorted(arr)) all_sorted = false;
    }
    out << n << "," << name << "," << (total_time / repeat) << "," << (all_sorted ? "Y" : "N") << "\n";
  };

  run_and_record("bitonic", [](vector<int>& a){ bitonic_sort_iterative_avx2_thread(a); }, n);
  run_and_record("qsort(stl)", [](vector<int>& a){ sort(a.begin(), a.end()); }, n);
  run_and_record("qsort_simple", [](vector<int>& a){ qsort_simple(a.data(), 0, a.size() - 1); }, n);
  run_and_record("qsort_avx256", [](vector<int>& a){ qsort_avx256(a.data(), 0, a.size() - 1); }, n);
  run_and_record("qsort_thread", [](vector<int>& a){ topknsort_qsort(a); }, n);
  run_and_record("msort_simple", [](vector<int>& a){ topknsort_msort_simple(a); }, n);
  run_and_record("msort_avx256", [](vector<int>& a){ topknsort_msort_avx256(a); }, n);
  run_and_record("msort_thread", [](vector<int>& a){ topknsort_msort(a); }, n);
  run_and_record("msort_thread_avx256", [](vector<int>& a){ topknsort_msort_thread_avx256(a); }, n);
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
  // std::string filename = "benchmark_" + get_timestamp() + ".csv";
  // std::ofstream fout(filename);
  // fout << "data_size,method,time(us),sorted\n";
  // test(2 << 12, fout);
  // test(2 << 13, fout);
  // test(2 << 14, fout);
  // test(2 << 15, fout);
  // test(2 << 16, fout);
  // test(2 << 17, fout);
  // test(2 << 18, fout);
  // test(2 << 19, fout);
  // test(2 << 20, fout);
  // test(2 << 21, fout);
  // fout.close();

  // std::string filename = "benchmark_topk1_" + get_timestamp() + ".csv";
  // std::ofstream fout(filename);
  // fout << "data_size,method,time(us),\n";
  // test_topk1(2 << 12, fout);
  // test_topk1(2 << 13, fout);
  // test_topk1(2 << 14, fout);
  // test_topk1(2 << 15, fout);
  // test_topk1(2 << 16, fout);
  // test_topk1(2 << 17, fout);
  // test_topk1(2 << 18, fout);
  // test_topk1(2 << 19, fout);
  // test_topk1(2 << 20, fout);
  // test_topk1(2 << 21, fout);
  // test_topk1(2 << 22, fout);
  // test_topk1(2 << 23, fout);
  // test_topk1(2 << 24, fout);
  // test_topk1(2 << 25, fout);
  // fout.close();

  std::string filename = "benchmark_topkn_" + get_timestamp() + "_thread" + to_string(THREAD_NUM) + ".csv";
  cout << filename << endl;
  std::ofstream fout(filename);
  fout << "data_size,method,time(us),sorted\n";
  test_topkn(2 << 12, fout);
  test_topkn(2 << 13, fout);
  test_topkn(2 << 14, fout);
  test_topkn(2 << 15, fout);
  test_topkn(2 << 16, fout);
  test_topkn(2 << 17, fout);
  test_topkn(2 << 18, fout);
  test_topkn(2 << 19, fout);
  test_topkn(2 << 20, fout);
  test_topkn(2 << 21, fout);
  fout.close();

  // int topk1 = 0;
  // int topk1_value = 0;
  // int n = 100000003;
  // vector<int> arr(n);
  // gen_arr(arr, n);
  // auto start = chrono::high_resolution_clock::now();
  // topk1int32simple(arr, topk1, topk1_value, true);
  // auto end = chrono::high_resolution_clock::now();
  // auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(max): " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple(arr, topk1, topk1_value, false);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(min): " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_avx256(arr, topk1, topk1_value, true);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(max) avx256: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_avx256(arr, topk1, topk1_value, false);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(min) avx256: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_thread(arr, topk1, topk1_value, true);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(max) thread: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_thread(arr, topk1, topk1_value, false);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(min) thread: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;
  
  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_thread_avx256(arr, topk1, topk1_value, true);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(max) thread+avx256: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;

  // start = chrono::high_resolution_clock::now();
  // topk1int32simple_thread_avx256(arr, topk1, topk1_value, false);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // cout << "Top 1(min) thread+avx256: " << topk1 << ", value: " << topk1_value << ", time(us): " << duration << endl;


  // int sort_n = 4194304;
  // bool sorted = false;
  // vector<int> sort_arr(sort_n);
  // gen_arr(sort_arr, sort_n);
  // vector<int> arr_copy;

  // arr_copy = sort_arr;
  // start = chrono::high_resolution_clock::now();
  // qsort_simple(arr_copy.data(), 0, sort_n - 1);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // sorted = check_sorted(arr_copy);
  // cout << "qsort_simple: n: "<< sort_n << ", time(us): " << duration << ", sorted: " << (sorted ? "Y" : "N") << endl;

  // arr_copy = sort_arr;
  // start = chrono::high_resolution_clock::now();
  // qsort_avx256(arr_copy.data(), 0, sort_n - 1);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // sorted = check_sorted(arr_copy);
  // cout << "qsort_avx256: n: "<< sort_n << ", time(us): " << duration << ", sorted: " << (sorted ? "Y" : "N") << endl;

  // arr_copy = sort_arr;
  // start = chrono::high_resolution_clock::now();
  // topknsort_qsort(arr_copy);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // sorted = check_sorted(arr_copy);
  // cout << "topknsort_qsort: n: "<< sort_n << ", time(us): " << duration << ", sorted: " << (sorted ? "Y" : "N") << endl;

  // arr_copy = sort_arr;
  // start = chrono::high_resolution_clock::now();
  // topknsort_msort(arr_copy);
  // end = chrono::high_resolution_clock::now();
  // duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
  // sorted = check_sorted(arr_copy);
  // cout << "topknsort_msort: n: "<< sort_n << ", time(us): " << duration << ", sorted: " << (sorted ? "Y" : "N") << endl;

  return 0;
}