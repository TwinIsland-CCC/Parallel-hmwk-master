
#ifndef ARG_MAX_FUSION_RAW_H_
#define ARG_MAX_FUSION_RAW_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <float.h>

#include "arg_min_max.h"
#include "nnacl_common.h"

#define CACHE_SIZE (8 * 0x10000)

#define DATA_TYPE int32_t
#define MIN_VALUE INT32_MIN
#define MAX_VALUE INT32_MAX
#define UNION_DATA i_data_
#define ARG_ELEMENT ArgElement32

int ArgCompareDescInt32(const void *a, const void *b) {                                                  
  DATA_TYPE b_value = ((ARG_ELEMENT *)b)->data_.UNION_DATA;                                                     
  DATA_TYPE a_value = ((ARG_ELEMENT *)a)->data_.UNION_DATA;                                                     
  if (b_value > a_value) {                                                                                       
    return 1;                                                                                                    
  }                                                                                                              
  if (b_value < a_value) {                                                                                       
    return -1;                                                                                                   
  }                                                                                                              
  return 0;                                                                                                      
}                                                                                                                
int ArgCompareAscInt32(const void *a, const void *b) {                                                   
  DATA_TYPE a_value = ((ARG_ELEMENT *)a)->data_.UNION_DATA;                                                     
  DATA_TYPE b_value = ((ARG_ELEMENT *)b)->data_.UNION_DATA;                                                     
  if (b_value > a_value) {                                                                                       
    return -1;                                                                                                   
  }                                                                                                              
  if (b_value < a_value) {                                                                                       
    return 1;                                                                                                    
  }                                                                                                              
  return 0;                                                                                                      
}                                                                                                                
                                                                                                                  
void ArgMinMaxTopK1Int32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                       
                            const ArgMinMaxComputeParam *param, int pre_axis_count, int axis_count,              
                            int after_axis_count) {                                                              
  bool out_value = param->out_value_;                                                                            
  DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                   
  int32_t *outputint = (int32_t *)output;                                                                        
  int get_max = param->get_max_ == true ? 1 : 0;
  int i = 0, j = 0, k = 0;                                                                                       
  for (i = 0; i < pre_axis_count; ++i) {                                                                         
    int output_offset = i * after_axis_count;                                                                    
    int input_offset = output_offset * axis_count;                                                               
    for (j = 0; j < after_axis_count; ++j) {                                                                     
      DATA_TYPE value = MIN_VALUE;                                                                               
      int index = 0;                                                                                             
      for (k = 0; k < axis_count; ++k) {                                                                         
        DATA_TYPE value_tmp = input[input_offset + k * after_axis_count + j];                                    
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
                                                                      
void ArgMinMaxDim0Int32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                   
                              const int32_t *in_shape, const ArgMinMaxComputeParam *param,                     
                              COMPARE_FUNCTION compare_func) {                                                 
  DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                 
  int32_t *outputint = (int32_t *)output;                                                                      
  int32_t i = 0;                                                                                               
  ARG_ELEMENT* element = (ARG_ELEMENT*) param->arg_elements_;                                                  
  int j = 0;                                                                                                   
  for (i = 0; i < param->in_strides_[0]; ++i) {                                                                
    for (j = 0; j < in_shape[0]; ++j) {                                                                        
      int offset = param->in_strides_[0] * j + i;                                                              
      element[j].index_ = (uint32_t)j;                                                                         
      element[j].data_.UNION_DATA = input[offset];                                                             
    }                                                                                                          
    qsort(element, in_shape[0], sizeof(ARG_ELEMENT), *compare_func);                                            
    for (j = 0; j < param->topk_; ++j) {                                                                       
      int out_offset = j * param->out_strides_[0] + i;                                                         
      if (param->out_value_) {                                                                                 
        outputfp32[out_offset] = element[j].data_.UNION_DATA;                                                  
      } else {                                                                                                 
        outputint[out_offset] = element[j].index_;                                                             
      }                                                                                                        
      if (output_value != NULL) {                                                                              
        output_value[out_offset] = element[j].data_.UNION_DATA;                                                
      }                                                                                                        
    }                                                                                                          
  }                                                                                                            
  return;                                                                                                      
}                                                                                                              
                                                                                                          
void ArgMinMaxDim1Int32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                   
                              const int32_t *in_shape, const ArgMinMaxComputeParam *param,                     
                              COMPARE_FUNCTION compare_func) {                                                 
  DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                 
  int32_t *outputint = (int32_t *)output;                                                                      
  int in_shape1 = in_shape[1];                                                                                 
  int i = 0, j = 0, k = 0;                                                                                     
  ARG_ELEMENT* element = (ARG_ELEMENT*) param->arg_elements_;                                                  
  for (i = 0; i < in_shape[0]; ++i) {                                                                          
    int in_dim0_offset = i * param->in_strides_[0];                                                            
    int out_dim0_offset = i * param->out_strides_[0];                                                          
    for (j = 0; j < param->in_strides_[1]; ++j) {                                                              
      for (k = 0; k < in_shape1; ++k) {                                                                        
        int offset = param->in_strides_[1] * k + in_dim0_offset + j;                                           
        element[k].index_ = (uint32_t)k;                                                                       
        element[k].data_.UNION_DATA = input[offset];                                                           
      }                                                                                                        
      qsort(element, in_shape1, sizeof(ARG_ELEMENT), *compare_func);                                            
      for (k = 0; k < param->topk_; ++k) {                                                                     
        int out_offset = out_dim0_offset + j + k * param->out_strides_[1];                                     
        if (param->out_value_) {                                                                               
          outputfp32[out_offset] = element[k].data_.UNION_DATA;                                                
        } else {                                                                                               
          outputint[out_offset] = element[k].index_;                                                           
        }                                                                                                      
        if (output_value != NULL) {                                                                            
          output_value[out_offset] = element[k].data_.UNION_DATA;                                              
        }                                                                                                      
      }                                                                                                        
    }                                                                                                          
  }                                                                                                            
  return;                                                                                                      
}                                                                                                              
                                                                                                          
void ArgMinMaxDim2Int32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                   
                              const int32_t *in_shape, const ArgMinMaxComputeParam *param,                     
                              COMPARE_FUNCTION compare_func) {                                                 
  int in_shape1 = in_shape[1];                                                                                 
  int in_shape2 = in_shape[2];                                                                                 
  DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                 
  int32_t *outputint = (int32_t *)output;                                                                      
  int i = 0, j = 0, k = 0, l = 0;                                                                              
  ARG_ELEMENT* element = (ARG_ELEMENT*) param->arg_elements_;                                                  
  for (i = 0; i < in_shape[0]; ++i) {                                                                          
    int in_dim0_offset = i * param->in_strides_[0];                                                            
    int out_dim0_offset = i * param->out_strides_[0];                                                          
    for (j = 0; j < in_shape1; ++j) {                                                                          
      int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;                                         
      int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;                                      
      for (k = 0; k < param->in_strides_[2]; ++k) {                                                            
        for (l = 0; l < in_shape2; ++l) {                                                                      
          int offset = param->in_strides_[2] * l + k + in_dim1_offset;                                         
          element[l].index_ = (uint32_t)l;                                                                     
          element[l].data_.UNION_DATA = input[offset];                                                         
        }                                                                                                      
        qsort(element, in_shape2, sizeof(ARG_ELEMENT), *compare_func);                                          
        for (l = 0; l < param->topk_; ++l) {                                                                   
          int out_offset = out_dim1_offset + k + l * param->out_strides_[2];                                   
          if (param->out_value_) {                                                                             
            outputfp32[out_offset] = element[l].data_.UNION_DATA;                                              
          } else {                                                                                             
            outputint[out_offset] = element[l].index_;                                                         
          }                                                                                                    
          if (output_value != NULL) {                                                                          
            output_value[out_offset] = element[l].data_.UNION_DATA;                                            
          }                                                                                                    
        }                                                                                                      
      }                                                                                                        
    }                                                                                                          
  }                                                                                                            
}                                                                                                              
                                                                                                          
void ArgMinMaxDim3Int32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                   
                              const int32_t *in_shape, const ArgMinMaxComputeParam *param,                     
                              COMPARE_FUNCTION compare_func) {                                                 
  int in_shape1 = in_shape[1];                                                                                 
  int in_shape2 = in_shape[2];                                                                                 
  int in_shape3 = in_shape[3];                                                                                 
  DATA_TYPE *outputfp32 = (DATA_TYPE *)output;                                                                 
  int32_t *outputint = (int32_t *)output;                                                                      
  int i = 0, j = 0, k = 0, l = 0;                                                                              
  ARG_ELEMENT* element = (ARG_ELEMENT*) param->arg_elements_;                                                  
  for (i = 0; i < in_shape[0]; ++i) {                                                                          
    int in_dim0_offset = i * param->in_strides_[0];                                                            
    int out_dim0_offset = i * param->out_strides_[0];                                                          
    for (j = 0; j < in_shape1; ++j) {                                                                          
      int in_dim1_offset = j * param->in_strides_[1] + in_dim0_offset;                                         
      int out_dim1_offset = j * param->out_strides_[1] + out_dim0_offset;                                      
      for (k = 0; k < in_shape2; ++k) {                                                                        
        int in_dim2_offset = k * param->in_strides_[2] + in_dim1_offset;                                       
        int out_dim2_offset = k * param->out_strides_[2] + out_dim1_offset;                                    
        for (l = 0; l < in_shape3; ++l) {                                                                      
          int offset = l + in_dim2_offset;                                                                     
          element[l].index_ = (uint32_t)l;                                                                     
          element[l].data_.UNION_DATA = input[offset];                                                         
        }                                                                                                      
        qsort(element, in_shape3, sizeof(ARG_ELEMENT), *compare_func);                                          
        for (l = 0; l < param->topk_; ++l) {                                                                   
          int out_offset = out_dim2_offset + l;                                                                
          if (param->out_value_) {                                                                             
            outputfp32[out_offset] = element[l].data_.UNION_DATA;                                              
          } else {                                                                                             
            outputint[out_offset] = (int)(element[l].index_);                                                  
          }                                                                                                    
          if (output_value != NULL) {                                                                          
            output_value[out_offset] = element[l].data_.UNION_DATA;                                            
          }                                                                                                    
        }                                                                                                      
      }                                                                                                        
    }                                                                                                          
  }                                                                                                            
}                                                                                                              
                                                                                                          
void ArgMinMaxInt32(const DATA_TYPE *input, void *output, DATA_TYPE *output_value,                   
                              const int32_t *in_shape, const ArgMinMaxComputeParam *param) {                   
  if (param->topk_ == 1) {                                                                                     
    int pre_axis_count = 1;                                                                                    
    int axis_count = 1;                                                                                        
    int after_axis_count = 1;                                                                                  
    ComputeAxisDims(in_shape, param->dims_size_, param->axis_, &pre_axis_count, &axis_count, &after_axis_count);                                                                                                                                                                              
    ArgMinMaxTopK1Int32(input, output, output_value, param, pre_axis_count, axis_count, after_axis_count);                                                                                                      
    return;                                                                                                    
  }                                                                                                            
                                                                                                          
  COMPARE_FUNCTION compare_function = NULL;                                                                    
  if (param->get_max_) {                                                                                       
    compare_function = ArgCompareDescInt32;                                                            
  } else {                                                                                                     
    compare_function = ArgCompareAscInt32;                                                             
  }                                                                                                            
                                                                                                          
  switch (param->axis_) {                                                                                      
    case 0:                                                                                                    
      ArgMinMaxDim0Int32(input, output, output_value, in_shape, param, compare_function);                
      break;                                                                                                   
    case 1:                                                                                                    
      ArgMinMaxDim1Int32(input, output, output_value, in_shape, param, compare_function);                
      break;                                                                                                   
    case 2:                                                                                                    
      ArgMinMaxDim2Int32(input, output, output_value, in_shape, param, compare_function);                
      break;                                                                                                   
    case 3:                                                                                                    
      ArgMinMaxDim3Int32(input, output, output_value, in_shape, param, compare_function);                
      break;                                                                                                   
  }                                                                                                            
  return;                                                                                                      
}                            


#undef DATA_TYPE
#undef MIN_VALUE
#undef MAX_VALUE
#undef UNION_DATA
#undef ARG_ELEMENT

#endif // ARG_MAX_FUSION_RAW_H_
