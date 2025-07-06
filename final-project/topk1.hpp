#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <float.h>

#include "arg_min_max.h"
#include "nnacl_common.h"

#include <vector>
#include <algorithm>

#include <immintrin.h>
#include "BS_thread_pool.hpp"

using namespace std;

void topk1Int32(const int *input, void *output, int *output_value,                       
                            const ArgMinMaxComputeParam *param, int pre_axis_count, int axis_count,              
                            int after_axis_count) {                                                              
  bool out_value = param->out_value_;                                                                            
  int *outputfp32 = (int *)output;                                                                   
  int32_t *outputint = (int32_t *)output;                                                                        
  int get_max = param->get_max_ == true ? 1 : 0;
  int i = 0, j = 0, k = 0;                                                                                       
  for (i = 0; i < pre_axis_count; ++i) {                                                                         
    int output_offset = i * after_axis_count;                                                                    
    int input_offset = output_offset * axis_count;                                                               
    for (j = 0; j < after_axis_count; ++j) {                                                                     
      int value = INT_MIN;                                                                               
      int index = 0;                                                                                             
      for (k = 0; k < axis_count; ++k) {                                                                         
        int value_tmp = input[input_offset + k * after_axis_count + j];                                    
        if ((value_tmp > value) == get_max) {                                                                                 
          value = value_tmp;                                                                                     
          index = k;                                                                                             
        }                                                                                                        
      }                                                                                                          
      if (out_value) {                                                                                           
        outputfp32[output_offset + j] = value;                                                                   
      } else {                                                                                                   
        outputint[output_offset + j] = index;                                                                    
      }                                                                                                          
      if (output_value != NULL) {                                                                                
        output_value[output_offset + j] = value;                                                                 
      }                                                                                                          
    }                                                                                                            
  }  
} 
  
void topk1int32simple(const vector<int>& input, int& output, int& output_value, bool get_max) {
  int n = input.size();
  int index = 0;
  int value = get_max ? INT_MIN : INT_MAX;
  for (int i = 0; i < n; ++i) {
    int value_tmp = input[i];                                    
    if ((value_tmp > value) == get_max) {                                                                                 
      value = value_tmp;                                                                                     
      index = i;                                                                                             
    }                                                    
  }
  output = index;
  output_value = value;
  return;
}


void topk1int32simple_avx256(const vector<int>& input, int& output, int& output_value, bool get_max) {
  int n = input.size();
  int value = get_max ? INT_MIN : INT_MAX;
  int index = 0;
  int i = 0;

  __m256i vb = get_max ? _mm256_set1_epi32(INT_MIN) : _mm256_set1_epi32(INT_MAX);
  __m256i vb_index = _mm256_set1_epi32(0);

  for (i = 0; i + 8 <= n; i += 8) {
    __m256i va = _mm256_loadu_si256((__m256i const*)(&input[i]));
    __m256i idxs = _mm256_set_epi32(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i);
    __m256i cmp = get_max ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
    vb = _mm256_blendv_epi8(vb, va, cmp);
    vb_index = _mm256_blendv_epi8(vb_index, idxs, cmp);
  }
  // 水平归约
  alignas(32) int buf[8], buf_index[8];
  _mm256_store_si256((__m256i*)buf, vb);
  _mm256_store_si256((__m256i*)buf_index, vb_index);
  for (int i = 0; i < 8; i++) {
    if ((buf[i] > value) == get_max) {
      value = buf[i];
      index = buf_index[i];
    }
  }
  // 处理剩余
  for (i; i < n; ++i) {
    int value_tmp = input[i];
    if ((value_tmp > value) == get_max) {
      value = value_tmp;
      index = i;
    }
  }
  output = index;
  output_value = value;
}

void topk1int32simple_thread(const vector<int>& input, int& output, int& output_value, bool get_max) {
  int n = input.size();
  int thread_num = thread::hardware_concurrency();
  int batch = (n + thread_num - 1) / thread_num;  // 向上取整除法
  vector<future<pair<int, int>>> results;
  BS::thread_pool pool(thread_num);

  for (int t = 0; t < thread_num; ++t) {
    int start = t * batch;
    int end = start + batch < n ? start + batch : n;
    if (start >= end) continue;
    results.push_back(pool.submit_task([&, start, end]() -> pair<int, int> {  // 有返回值的时候使用submit
      int local_index = start;
      int local_value = get_max ? INT_MIN : INT_MAX;
      for (int i = start; i < end; ++i) {
        int value_tmp = input[i];
        if ((value_tmp > local_value) == get_max) {
          local_value = value_tmp;
          local_index = i;
        }
      }
      return {local_index, local_value};
    }));
  }
  int index = 0;
  int value = get_max ? INT_MIN : INT_MAX;
  for (auto& f : results) {
    auto [local_index, local_value] = f.get();
    if ((local_value > value) == get_max) {
      value = local_value;
      index = local_index;
    }
  }
  output = index;
  output_value = value;
}

void topk1int32simple_thread_avx256(const vector<int>& input, int& output, int& output_value, bool get_max) {
  int n = input.size();
  int thread_num = thread::hardware_concurrency();
  int batch = (n + thread_num - 1) / thread_num;  // 向上取整除法
  vector<future<pair<int, int>>> results;
  BS::thread_pool pool(thread_num);

  for (int t = 0; t < thread_num; ++t) {
    int start = t * batch;
    int end = start + batch < n ? start + batch : n;
    if (start >= end) continue;
    results.push_back(pool.submit_task([&, start, end]() -> pair<int, int> {  // 有返回值的时候使用submit
      int local_index = start;
      int local_value = get_max ? INT_MIN : INT_MAX;
      __m256i vb = get_max ? _mm256_set1_epi32(INT_MIN) : _mm256_set1_epi32(INT_MAX);
      __m256i vb_index = _mm256_set1_epi32(start);
      int i = start;
      for (i = start; i + 8 <= end; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(&input[i]));
        __m256i idxs = _mm256_set_epi32(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i);
        __m256i cmp = get_max ? _mm256_cmpgt_epi32(va, vb) : _mm256_cmpgt_epi32(vb, va);
        vb = _mm256_blendv_epi8(vb, va, cmp);
        vb_index = _mm256_blendv_epi8(vb_index, idxs, cmp);
      }
      // 水平归约
      alignas(32) int buf[8], buf_index[8];
      _mm256_store_si256((__m256i*)buf, vb);
      _mm256_store_si256((__m256i*)buf_index, vb_index);
      for (int i = 0; i < 8; i++) {
        if ((buf[i] > local_value) == get_max) {
          local_value = buf[i];
          local_index = buf_index[i];
        }
      }
      // 处理剩余
      for (i; i < end; ++i) {
        int value_tmp = input[i];
        if ((value_tmp > local_value) == get_max) {
          local_value = value_tmp;
          local_index = i;
        }
      }
      output = local_index;
      output_value = local_value;
      return {local_index, local_value};
    }));
  }
  int index = 0;
  int value = get_max ? INT_MIN : INT_MAX;
  for (auto& f : results) {
    auto [local_index, local_value] = f.get();
    if ((local_value > value) == get_max) {
      value = local_value;
      index = local_index;
    }
  }
  output = index;
  output_value = value;
}