// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>


//!see https://zhuanlan.zhihu.com/p/80902998


int const threadsPerBlock = sizeof(unsigned long long) * 8;//分块数量

//在gpu上计算IOU
__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;//确定当前block的横纵坐标

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  ////求当前block的行长度，如果最后一个block不够除，则取余下的，
  //比如ceil(105/25) = 5，105 = 4 * 25 + 5最后一块高为5，此时row_size=5，其余的row_size = 25

  // 共享内存，加速数据读取，
  //同一个block有共享内存，所以先使用共享内存存下当前block全部需要读取的数据
  //(即box的坐标和置信度)然后就不在dev_boxes里面读数据了，而是读share memory里面的数据
  __shared__ float block_boxes[threadsPerBlock * 5];

  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }

  //为了保证线程安全，必须等所有的线程都把数据存到share memory以后，统一开始线程
  __syncthreads();
  // 这个if判断去掉多余的thread，保证余下的块可以被正确执行
  // 每个block里面有row_size个线程
  // 每个线程i，for一个col_size的循环，计算该block里面第i个box和该block中每个列box的IOU
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;//对角线上的block, //自己跟自己就不要计算IOU了
    }
    for (i = start; i < col_size; i++) {
       //主循环，求该box和所有列box的IOU，如果满足条件，则使用一个mask把该位置1
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;//掩码
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {

  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(boxes.device());

  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 4);//tensor.select(1, index)等效于tensor[:, index]
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));


  // 定义blocks的数量和每个block的线程数
  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
   // 调用kernel，最后在mask_dev中求出每两个框的IoU是否超过阈值t
  nms_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpyAsync(
			  &mask_host[0],
			  mask_dev,
			  sizeof(unsigned long long) * boxes_num * col_blocks,
			  cudaMemcpyDeviceToHost,
			  at::cuda::getCurrentCUDAStream()
			  ));

  std::vector<unsigned long long> remv(col_blocks);// 初始是所有框都在S里面，移出标记都置为0
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);// 初始是所有框都在S里面，移出标记都置为0

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock; //求这个box是在哪个block里面计算的
    int inblock = i % threadsPerBlock; //求这个box在block的哪个线程计算的
    
    // 对于每个box，如果他在S中，则加入结果集，并移出S
    // 并把和他的IOU大于阈值的所有box全部移出S
    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;//加入结果集操作
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];//移出S操作
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}
