#define SIMULATOR

#ifdef SIMULATOR
#include<simulator/initial.h>
#include<simulator/DMA.h>
#else
#include<78NE/initial.h>
#include<78NE/DMA.h>
#endif

#include <stdio.h>
#include<time.h>
#include<stdlib.h>
#include<stdint.h>
#include <csl/csl_cache.h>
#include <csl/csl_cacheAux.h>
#include <tistdtypes.h>
#include <inttypes.h>

#include <math.h>

#include "external_sort.h"

static void merge_two_blocks_Fp32(ArgElement32 *src, ArgElement32 *dst, int left_start, int left_size, int right_start, int right_size, void *cache, unsigned cache_size)
{
    int elem_bytes = sizeof(ArgElement32);
    int M = cache_size / (elem_bytes * 3);
    if (M < 1)
        return;
    char *cptr = (char *)cache;
    ArgElement32 *bufA_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    ArgElement32 *bufB_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    ArgElement32 *out_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    int a_ext_pos = left_start, b_ext_pos = right_start;
    int a_remain = left_size, b_remain = right_size;
    int a_buf_cnt = 0, a_buf_pos = 0;
    int b_buf_cnt = 0, b_buf_pos = 0;
    int out_cnt = 0;
    int write_pos = left_start;
    while ((a_remain > 0 || a_buf_pos < a_buf_cnt) || (b_remain > 0 || b_buf_pos < b_buf_cnt))
    {
        if (a_buf_pos == a_buf_cnt && a_remain > 0)
        {
            int cnt = (a_remain < M ? a_remain : M);
            my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + a_ext_pos), (Uint32)bufA_vals);
            a_ext_pos += cnt;
            a_remain -= cnt;
            a_buf_cnt = cnt;
            a_buf_pos = 0;
        }
        if (b_buf_pos == b_buf_cnt && b_remain > 0)
        {
            int cnt = (b_remain < M ? b_remain : M);
            my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + b_ext_pos), (Uint32)bufB_vals);
            b_ext_pos += cnt;
            b_remain -= cnt;
            b_buf_cnt = cnt;
            b_buf_pos = 0;
        }
        if (a_buf_pos < a_buf_cnt && b_buf_pos < b_buf_cnt)
        {
            if (bufA_vals[a_buf_pos].data_.f_data_ <= bufB_vals[b_buf_pos].data_.f_data_)
            {
                out_vals[out_cnt] = bufA_vals[a_buf_pos];
            }
            else
            {
                out_vals[out_cnt] = bufB_vals[b_buf_pos];
            }
        }
        else if (a_buf_pos < a_buf_cnt)
        {
            out_vals[out_cnt] = bufA_vals[a_buf_pos];
        }
        else
        {
            out_vals[out_cnt] = bufB_vals[b_buf_pos];
        }
        out_cnt++;
        if (out_cnt == M)
        {
            my_dma_trans(out_cnt * sizeof(ArgElement32), (Uint32)out_vals, (Uint32)(dst + write_pos));
            write_pos += out_cnt;
            out_cnt = 0;
        }
    }
    if (out_cnt > 0)
    {
        my_dma_trans(out_cnt * sizeof(ArgElement32), (Uint32)out_vals, (Uint32)(dst + write_pos));
    }
}

static void merge_two_blocks_Int32(ArgElement32 *src, ArgElement32 *dst, int left_start, int left_size, int right_start, int right_size, void *cache, unsigned cache_size)
{
    int elem_bytes = sizeof(ArgElement32);
    int M = cache_size / (elem_bytes * 3);
    if (M < 1)
        return;
    char *cptr = (char *)cache;
    ArgElement32 *bufA_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    ArgElement32 *bufB_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    ArgElement32 *out_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    int a_ext_pos = left_start, b_ext_pos = right_start;
    int a_remain = left_size, b_remain = right_size;
    int a_buf_cnt = 0, a_buf_pos = 0;
    int b_buf_cnt = 0, b_buf_pos = 0;
    int out_cnt = 0;
    int write_pos = left_start;
    while ((a_remain > 0 || a_buf_pos < a_buf_cnt) || (b_remain > 0 || b_buf_pos < b_buf_cnt))
    {
        if (a_buf_pos == a_buf_cnt && a_remain > 0)
        {
            int cnt = (a_remain < M ? a_remain : M);
            my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + a_ext_pos), (Uint32)bufA_vals);
            a_ext_pos += cnt;
            a_remain -= cnt;
            a_buf_cnt = cnt;
            a_buf_pos = 0;
        }
        if (b_buf_pos == b_buf_cnt && b_remain > 0)
        {
            int cnt = (b_remain < M ? b_remain : M);
            my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + b_ext_pos), (Uint32)bufB_vals);
            b_ext_pos += cnt;
            b_remain -= cnt;
            b_buf_cnt = cnt;
            b_buf_pos = 0;
        }
        if (a_buf_pos < a_buf_cnt && b_buf_pos < b_buf_cnt)
        {
            if (bufA_vals[a_buf_pos].data_.i_data_ <= bufB_vals[b_buf_pos].data_.i_data_)
            {
                out_vals[out_cnt] = bufA_vals[a_buf_pos];
            }
            else
            {
                out_vals[out_cnt] = bufB_vals[b_buf_pos];
            }
        }
        else if (a_buf_pos < a_buf_cnt)
        {
            out_vals[out_cnt] = bufA_vals[a_buf_pos];
        }
        else
        {
            out_vals[out_cnt] = bufB_vals[b_buf_pos];
        }
        out_cnt++;
        if (out_cnt == M)
        {
            my_dma_trans(out_cnt * sizeof(ArgElement32), (Uint32)out_vals, (Uint32)(dst + write_pos));
            write_pos += out_cnt;
            out_cnt = 0;
        }
    }
    if (out_cnt > 0)
    {
        my_dma_trans(out_cnt * sizeof(ArgElement32), (Uint32)out_vals, (Uint32)(dst + write_pos));
    }
}
static void swap_Fp32(ArgElement32 *a, ArgElement32 *b)
{
    ArgElement32 tmp_v = *a;
    *a = *b;
    *b = tmp_v;
}
void myqsort_Fp32(ArgElement32 *input, int left, int right)
{
    if (left >= right)
        return;
    ArgElement32 pivot = input[right];
    int i = left - 1;
    int j;
    for (j = left; j < right; ++j)
    {
        if (input[j].data_.f_data_ <= pivot.data_.f_data_)
        {
            ++i;
            swap_Fp32(&input[i], &input[j]);
        }
    }
    swap_Fp32(&input[i + 1], &input[right]);
    myqsort_Fp32(input, left, i);
    myqsort_Fp32(input, i + 2, right);
}
static void swap_Int32(ArgElement32 *a, ArgElement32 *b)
{
    ArgElement32 tmp_v = *a;
    *a = *b;
    *b = tmp_v;
}
void myqsort_Int32(ArgElement32 *input, int left, int right)
{
    if (left >= right)
        return;
    ArgElement32 pivot = input[right];
    int i = left - 1;
    int j;
    for (j = left; j < right; ++j)
    {
        if (input[j].data_.i_data_ <= pivot.data_.i_data_)
        {
            ++i;
            swap_Int32(&input[i], &input[j]);
        }
    }
    swap_Int32(&input[i + 1], &input[right]);
    myqsort_Int32(input, left, i);
    myqsort_Int32(input, i + 2, right);
}

void non_recursive_qsort_int32(int *arr, int left, int right) {
    if (left >= right) return;
    typedef struct { int l, r; } StackNode;
    StackNode stack[64];
    int top = 0;
    stack[top++] = (StackNode){left, right};

    while (top > 0) {
        StackNode node = stack[--top];
        int l = node.l, r = node.r;
        if (l >= r) continue;

        int pivot = arr[r];
        int i = l - 1, j = l;
        for (j = l; j < r; ++j) {
            if (arr[j] < pivot) {
                ++i;
                int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
        }
        int mid = i + 1;
        int tmp = arr[mid]; arr[mid] = arr[r]; arr[r] = tmp;

        if (mid + 1 < r) stack[top++] = (StackNode){mid + 1, r};
        if (l < mid - 1)  stack[top++] = (StackNode){l, mid - 1};
    }
}

void external_sort_Fp32(ArgElement32 *src, ArgElement32 *dst, int length, void *cache, unsigned cache_size)
{
    if (length <= 1)
    {
        if (src != dst)
        {
            my_dma_trans(length * sizeof(ArgElement32), (Uint32)(src), (Uint32)(dst));
        }
        return;
    }
    int elem_bytes = sizeof(ArgElement32);
    int M = cache_size / elem_bytes;
    if (M < 1)
        return;
    char *cptr = (char *)cache;
    ArgElement32 *buf_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    int start;
    for (start = 0; start < length; start += M)
    {
        int cnt = (length - start < M ? length - start : M);
        my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + start), (Uint32)buf_vals);
        non_recursive_qsort_int32(buf_vals, 0, cnt - 1);
        my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)buf_vals, (Uint32)(dst + start));
    }
    ArgElement32 *cur_src = dst;
    ArgElement32 *cur_dst = src;
    int width = M;
    int rounds = 0;
    while (width < length)
    {
        int left;
        for (left = 0; left < length; left += 2 * width)
        {
            int left_size = (length - left < width ? length - left : width);
            int right = left + width;
            int right_size = (length - right < width ? length - right : width);
            if (right >= length)
            {
                my_dma_trans(left_size * sizeof(ArgElement32), (Uint32)(cur_src + left), (Uint32)(cur_dst + left));
            }
            else
            {
                merge_two_blocks_Fp32(cur_src, cur_dst, left, left_size, right, right_size, cache, cache_size);
            }
        }
        ArgElement32 *tmp_vals = cur_src;
        cur_src = cur_dst;
        cur_dst = tmp_vals;
        width <<= 1;
        rounds++;
    }
    if (rounds & 1)
    {
        my_dma_trans(length * sizeof(ArgElement32), (Uint32)cur_src, (Uint32)dst);
    }
}
void external_sort_Int32(ArgElement32 *src, ArgElement32 *dst, int length, void *cache, unsigned cache_size)
{
    if (length <= 1)
    {
        if (src != dst)
        {
            my_dma_trans(length * sizeof(ArgElement32), (Uint32)(src), (Uint32)(dst));
        }
        return;
    }
    int elem_bytes = sizeof(ArgElement32);
    int M = cache_size / elem_bytes;
    if (M < 1)
        return;
    char *cptr = (char *)cache;
    ArgElement32 *buf_vals = (ArgElement32 *)cptr;
    cptr += M * sizeof(ArgElement32);
    int start;
    for (start = 0; start < length; start += M)
    {
        int cnt = (length - start < M ? length - start : M);
        my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)(src + start), (Uint32)buf_vals);
        myqsort_Int32(buf_vals, 0, cnt - 1);
        my_dma_trans(cnt * sizeof(ArgElement32), (Uint32)buf_vals, (Uint32)(dst + start));
    }
    ArgElement32 *cur_src = dst;
    ArgElement32 *cur_dst = src;
    int width = M;
    int rounds = 0;
    while (width < length)
    {
        int left;
        for (left = 0; left < length; left += 2 * width)
        {
            int left_size = (length - left < width ? length - left : width);
            int right = left + width;
            int right_size = (length - right < width ? length - right : width);
            if (right >= length)
            {
                my_dma_trans(left_size * sizeof(ArgElement32), (Uint32)(cur_src + left), (Uint32)(cur_dst + left));
            }
            else
            {
                merge_two_blocks_Int32(cur_src, cur_dst, left, left_size, right, right_size, cache, cache_size);
            }
        }
        ArgElement32 *tmp_vals = cur_src;
        cur_src = cur_dst;
        cur_dst = tmp_vals;
        width <<= 1;
        rounds++;
    }
    if (rounds & 1)
    {
        my_dma_trans(length * sizeof(ArgElement32), (Uint32)cur_src, (Uint32)dst);
    }
}
