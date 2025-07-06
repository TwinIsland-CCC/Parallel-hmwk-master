/*
 * arg_min_max.h
 *
 *  Created on: 2025年4月27日
 *      Author: CCC
 */

#ifndef ARG_MIN_MAX_H_
#define ARG_MIN_MAX_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <float.h>

#include "op_base.h"

typedef uint16_t float16_t;

typedef struct ArgMinMaxParameter {
  OpParameter op_parameter_;
  int32_t axis_;
  int32_t topk_;
  bool keep_dims_;
  bool out_value_;
} ArgMinMaxParameter;

typedef struct ArgElement32 {
  uint32_t index_;
  union ArgData32 {
    int32_t i_data_;
    float f_data_;
  } data_;
} ArgElement32;

typedef struct ArgElement16 {
  uint32_t index_;
  union ArgData16 {
    int16_t i16_data_;
    float16_t f16_data_;
// #ifdef ENABLE_ARM
// #if (!SUPPORT_NNIE) || (defined SUPPORT_34XX)
//     float16_t f16_data_;
// #endif
// #endif
  } data_;
} ArgElement16;

typedef struct ArgElement8 {
  uint32_t index_;
  union ArgData8 {
    int8_t i8_data_;
  } data_;
} ArgElement8;

typedef struct ArgElement64 {
  uint32_t index_;
  union ArgData64 {
    int8_t i8_data_;
    int16_t i16_data_;
    float16_t f16_data_;
    int32_t i_data_;
    float f_data_;
// #ifdef ENABLE_ARM
// #if (!SUPPORT_NNIE) || (defined SUPPORT_34XX)
//     float16_t f16_data_;
// #endif
// #endif
  } data_;
} ArgElement64;

typedef struct ArgElement128 {
  uint32_t index_;
  union ArgData128 {
    int8_t i8_data_;
    int16_t i16_data_;
    float16_t f16_data_;
    int32_t i_data_;
    float f_data_;
// #ifdef ENABLE_ARM
// #if (!SUPPORT_NNIE) || (defined SUPPORT_34XX)
//     float16_t f16_data_;
// #endif
// #endif
  } data_;
} ArgElement128;

typedef int (*COMPARE_FUNCTION)(const void *a, const void *b);

typedef struct ArgMinMaxComputeParam {
  int32_t axis_;
  int32_t dims_size_;
  int32_t topk_;
  bool get_max_;  // 用来标识使用argmax还是min。1是max
  bool keep_dims_;  // 保持维度
  bool out_value_;  // 是否输出索引对应值
  int32_t in_strides_[COMM_SHAPE_SIZE];
  int32_t out_strides_[COMM_SHAPE_SIZE];
  void *arg_elements_;
} ArgMinMaxComputeParam;

typedef struct ArgMinMaxStruct {
  ArgMinMaxComputeParam compute_;
  bool arg_elements_alloc_;
} ArgMinMaxStruct;


#endif /* ARG_MIN_MAX_H_ */
