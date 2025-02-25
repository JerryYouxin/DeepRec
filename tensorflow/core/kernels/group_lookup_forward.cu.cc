/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include <inttypes.h>

#include <exception>
#include <string>

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename TKey, typename TValue>
struct GroupEmbeddingForWardArgs {
  int nnz_;
  int64_t emb_row_size_;
  TKey *sp_values_;
  int64_t *sp_indices_;
  int *offset_indices_;
  TValue *emb_variable_;
  TValue *emb_vector_;
};

template <typename TKey, typename TValue>
__global__ void SetToIntMaxSTG128(GroupEmbeddingForWardArgs<TKey, TValue> *args,
                                  const int batch_size, const int num_lookups) {
  const int thread_offset = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int int_max = 0x7fffffff;
  for (int idx = 0; idx < num_lookups; ++idx) {
    int *values_offset = args[idx].offset_indices_;
    if (thread_offset + 4 < batch_size) {
      int4 four = make_int4(int_max, int_max, int_max, int_max);
      *((int4 *)(values_offset + thread_offset)) = four;
    } else if (thread_offset < batch_size) {
      for (int i = thread_offset; i < batch_size; i++) {
        values_offset[i] = int_max;
      }
    }
  }
}

__global__ void CalcPerElementRowInBatchValuesOffset(const int64_t *indices,
                                                     int *values_offset,
                                                     const int64_t nnz) {
  const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_offset < int(nnz)) {
    const int64_t element_row = indices[2 * thread_offset];
    atomicMin(values_offset + int(element_row), thread_offset);
  }
}

template <typename TKey, typename TValue>
__global__ void RemoveBadCase(GroupEmbeddingForWardArgs<TKey, TValue> *args,
                              const int64_t batch_size, const int num_lookups) {
  const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int int_max = 0x7fffffff;
  if (thread_offset < int(batch_size - 1)) {
    for (int i = 0; i < num_lookups; ++i) {
      volatile int *values_offset = args[i].offset_indices_;
      while (values_offset[thread_offset + 1] == int_max) {
      }
      const int compare = values_offset[thread_offset + 1];
      atomicMin((int *)values_offset + int(thread_offset), compare);
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner>
__global__ void ComputeEVFn(const int batch_size, const int dimension,
                            const float max_norm, const int num_lookups,
                            GroupEmbeddingForWardArgs<TKey, TValue> *args) {
  __shared__ float l2_sum[1];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid < batch_size && tid < dimension) {
    for (size_t i = 0; i < num_lookups; ++i) {
      int value_offset = args[i].offset_indices_[bid];
      int feature_num;
      if (bid == int(batch_size) - 1) {
        feature_num = int(args[i].nnz_) - value_offset;
      } else {
        feature_num = args[i].offset_indices_[bid + 1] - value_offset;
      }

      TValue out = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; ++j) {
        int feature_offset = (value_offset + j) * dimension;
        TValue sum = args[i].emb_variable_[feature_offset + tid];
        if (max_norm >= 0.0f) {
          if (tid == 0) {
            l2_sum[0] = 0.0f;
          }
          __syncthreads();
          atomicAdd(l2_sum, sum * sum);
          __syncthreads();
          float l2_norm = sqrtf(l2_sum[0]);
          if (l2_norm > max_norm) {
            sum *= max_norm / l2_norm;
          }
        }
        out += sum;
      }
      out = Combine<combiner>(out, feature_num);
      args[i].emb_vector_[bid * dimension + tid] = out;
    }
  }
}

template <typename TKey, typename TValue, Combiner combiner>
__global__ void ComputeSparseFn(const int batch_size, const int emb_vec_size,
                                const float max_norm, const int num_lookups,
                                GroupEmbeddingForWardArgs<TKey, TValue> *args) {
  __shared__ float l2_sum[1];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid < batch_size && tid < emb_vec_size) {
    for (int idx = 0; idx < num_lookups; ++idx) {
      int value_offset = args[idx].offset_indices_[bid];
      int feature_num;
      if (bid == int(batch_size) - 1) {
        feature_num = int(args[idx].nnz_) - value_offset;
      } else {
        feature_num = args[idx].offset_indices_[bid + 1] - value_offset;
      }
      TValue out = 0.0f;
      const TValue *emb_variable = args[idx].emb_variable_;
      const int64_t emb_dim_limit = args[idx].emb_row_size_;
      for (int i = 0; i < feature_num; i++) {
        int64_t indices = int(args[idx].sp_values_[value_offset + i]);
        TValue emb_element = 0.0;
        if (FastBoundsCheck(indices, emb_dim_limit)) {
          emb_element = emb_variable[indices * emb_vec_size + tid];
        }
        if (max_norm >= 0.0f) {
          // calc l2 norm of this emb row(per block) and compare with max_norm.
          // if greater than max_norm, then clip every element with factor
          // max_norm / l2norm
          if (tid == 0) {
            l2_sum[0] = 0.0f;
          }
          __syncthreads();
          atomicAdd(l2_sum, emb_element * emb_element);
          __syncthreads();
          float l2_norm = sqrtf(l2_sum[0]);
          if (l2_norm > max_norm) {
            emb_element *= max_norm / l2_norm;
          }
        }
        out += emb_element;
      }

      out = Combine<combiner>(out, feature_num);
      args[idx].emb_vector_[bid * emb_vec_size + tid] = out;
    }
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingForWard {
 public:
  void initialize(int num_lookups) {
    nums = num_lookups;
    args_.resize(num_lookups);
    CK_CUDA_THROW_(cudaMalloc(
        &d_args_,
        num_lookups * sizeof(GroupEmbeddingForWardArgs<TKey, TValue>)));
  }

  ~GroupEmbeddingForWard() {
    if (d_args_) {
      CK_CUDA_THROW_(cudaFree(d_args_));
    }
  }

  void set(int idx, TValue *emb_variable, TValue *emb_vector,
           int *offset_indices, int nnz, int64_t *sp_indices = nullptr,
           TKey *sp_values = nullptr, int64_t emb_row_size = -1) {
    args_[idx].emb_variable_ = emb_variable;
    args_[idx].emb_vector_ = emb_vector;
    args_[idx].offset_indices_ = offset_indices;
    args_[idx].sp_indices_ = sp_indices;
    args_[idx].nnz_ = nnz;
    args_[idx].sp_values_ = sp_values;
    args_[idx].emb_row_size_ = emb_row_size;
  }

  template <typename GradFn>
  void compute(GradFn fn, const int batch_size, const int dimension,
               const float max_norm, cudaStream_t stream) {
    CK_CUDA_THROW_(cudaMemcpyAsync(
        d_args_, args_.data(),
        args_.size() * sizeof(GroupEmbeddingForWardArgs<TKey, TValue>),
        cudaMemcpyHostToDevice, stream));

    {
      const int threads = 1024;
      int blocks = batch_size / threads;
      blocks = batch_size % threads == 0 ? blocks : blocks + 1;
      SetToIntMaxSTG128<<<blocks, threads, 0, stream>>>(d_args_,
                                                        int(batch_size), nums);
    }
    {
      for (int i = 0; i < nums; ++i) {
        const int threads = 1024;
        int &nnz = args_[i].nnz_;
        int blocks = nnz % threads == 0 ? (nnz / threads) : (nnz / threads + 1);
        // calculate values offset
        CalcPerElementRowInBatchValuesOffset<<<blocks, threads, 0, stream>>>(
            args_[i].sp_indices_, args_[i].offset_indices_, nnz);
      }
    }
    {
      const int threads = 1024;
      int blocks = batch_size % threads == 0 ? (batch_size / threads)
                                             : (batch_size / threads + 1);

      RemoveBadCase<<<blocks, threads, 0, stream>>>(d_args_, batch_size, nums);
    }
    CK_CUDA_THROW_(cudaStreamSynchronize(stream));
    {
      // future improve to mapping tp batch_size * num_lookup
      const int block_size = int(batch_size);
      const int threads = int(dimension);
      fn<<<block_size, threads, 0, stream>>>(batch_size, dimension, max_norm,
                                             nums, d_args_);
    }

    CK_CUDA_THROW_(cudaGetLastError());
  }

 protected:
  int nums;
  std::vector<GroupEmbeddingForWardArgs<TKey, TValue>> args_;
  GroupEmbeddingForWardArgs<TKey, TValue> *d_args_;
};

template <typename TKey, typename TValue>
class GroupLookupForWardBaseOp : public OpKernel {
 public:
  explicit GroupLookupForWardBaseOp(OpKernelConstruction *c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    lookuper_.initialize(num_lookups_);
  }

 protected:
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
  GroupEmbeddingForWard<TKey, TValue> lookuper_;
};

template <typename TFKey, typename TKey, typename TValue>
class MultiKvResourceGatherOp : public GroupLookupForWardBaseOp<TKey, TValue> {
 public:
  explicit MultiKvResourceGatherOp(OpKernelConstruction *c)
      : GroupLookupForWardBaseOp<TKey, TValue>(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &is_use_default_value_tensor_));
    if (is_use_default_value_tensor_) {
      get_default_v_fn_ = [](TValue *default_v, TFKey id, int64 index,
                             int64 total_dim,
                             int64 len) { return default_v + len * index; };
    } else {
      get_default_v_fn_ = [](TValue *default_v, TFKey id, int64 index,
                             int64 total_dim, int64 len) {
        return default_v + len * (id % total_dim);
      };
    }
  }

  ~MultiKvResourceGatherOp() { delete[] occupy_flag_; }

  void Compute(OpKernelContext *ctx) override {
    EmbeddingVar<TFKey, TValue> *ev = nullptr;
    const auto &device = ctx->eigen_device<GPUDevice>();
    int64_t batch_size = -1;
    TValue *default_v = nullptr;

    for (size_t i = 0; i < this->num_lookups_; ++i) {
      const Tensor &sp_values_tensor = ctx->input(this->num_lookups_ + i);
      auto sp_values = sp_values_tensor.flat<TFKey>();
      int64 N = sp_values_tensor.NumElements();

      const Tensor &sp_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      auto sp_indices = sp_indices_tensor.flat<int64>().data();
      int nnz = sp_indices_tensor.shape().dim_size(0);

      const Tensor &dense_shape_tensor = ctx->input(this->num_lookups_ * 3 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();

      if (batch_size == -1) {
        batch_size = dense_shape[0];
      }

      OP_REQUIRES(
          ctx, batch_size == dense_shape[0],
          errors::InvalidArgument(
              "shape[0] of each tensor in offset_indices are different."));

      OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, i), &ev));
      core::ScopedUnref unref_me(ev);
      if (is_use_default_value_tensor_) {
        default_v = (TValue *)ctx->input(4 * this->num_lookups_).data();
      } else {
        default_v = ev->GetDefaultValuePtr();
      }
      // DEBUG
      int64 dimension = ev->ValueLen();
      // DEBUG
      const TFKey *key_base = sp_values.data();
      Tensor out_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TValue>::value,
                                             {N * dimension}, &out_tensor));
      TValue *out_base = out_tensor.flat<TValue>().data();

      if (ev->IsSingleHbm()) {
        if (is_use_default_value_tensor_) {
          Tensor default_values(ctx->input(4 * this->num_lookups_));
          auto default_value_num = default_values.NumElements() / dimension;
          auto default_values_matrix =
              default_values.shaped<TValue, 2>({default_value_num, dimension});
          TValue *default_v_base = &default_values_matrix(0, 0);
          ev->LookupOrCreate(key_base, out_base, default_v_base,
                             default_value_num, is_use_default_value_tensor_, N,
                             device);
        } else {
          ev->LookupOrCreate(key_base, out_base, ev->GetDefaultValuePtr(),
                             ev->GetDefaultValueDim(), true, N, device);
        }
      } else {
        auto out_flat =
            out_tensor.shaped<TValue, 2>({N, out_tensor.NumElements() / N});
        const int64 slice_elems = out_flat.dimension(1);
        const size_t slice_bytes = slice_elems * sizeof(TValue);
        TValue **memcpy_address = new TValue *[N];
        TFKey *indices_host = new TFKey[N];

        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int64 num_threads = worker_threads->num_threads;
        if (occupy_flag_ == nullptr) {
          mutex_lock l(m_init_occupy_flag_);
          // double check
          if (occupy_flag_ == nullptr) {
            occupy_flag_ = new bool[num_threads];
            memset(occupy_flag_, 0, sizeof(bool) * num_threads);
          }
        }
        std::vector<std::list<int64>> init_cursor_list(num_threads + 1);
        std::vector<std::list<int64>> copyback_cursor_list(num_threads + 1);

        volatile bool is_cpu_indices_ready = false;
        // Copy ids from GPU to CPU for CPU Lookup.
        auto stream = ctx->op_device_context()->stream();
        auto event_mgr = ctx->device()->tensorflow_gpu_device_info()->event_mgr;

        se::DeviceMemoryBase gpu_src(const_cast<TFKey *>(key_base),
                                     N * sizeof(TFKey));
        stream->ThenMemcpy(indices_host, gpu_src, N * sizeof(TFKey));
        SyncWithEventMgr(stream, event_mgr);

        uint64 main_thread_id = Env::Default()->GetCurrentThreadId();
        auto do_work = [this, indices_host, out_base, slice_elems, ctx, ev,
                        memcpy_address, &init_cursor_list,
                        &copyback_cursor_list, main_thread_id,
                        num_threads](int64 start, int64 limit) {
          uint64 thread_id = Env::Default()->GetCurrentThreadId();
          int64 position;
          if (thread_id == main_thread_id) {
            position = num_threads;
          } else {
            position = -1;
            {
              spin_rd_lock l(mu_);
              auto iter = hash_map_.find(thread_id);
              if (iter != hash_map_.end()) {
                position = iter->second;
              }
            }

            if (position == -1) {
              // bind a new thread to a local cursor_list
              position = thread_id % num_threads;
              while (!__sync_bool_compare_and_swap(&(occupy_flag_[position]),
                                                   false, true)) {
                position = (position + 1) % num_threads;
              }
              {
                spin_wr_lock l(mu_);
                hash_map_.insert(std::pair<uint64, int64>(thread_id, position));
              }
            }
          }
          ev->LookupWithFreqBatch(indices_host, memcpy_address, start, limit,
                                  init_cursor_list[position],
                                  copyback_cursor_list[position]);
        };
        Shard(num_threads, worker_threads->workers, N, slice_bytes, do_work);
        for (int i = 1; i < num_threads + 1; i++) {
          if (init_cursor_list[i].size() > 0) {
            init_cursor_list[0].splice(init_cursor_list[0].end(),
                                       init_cursor_list[i]);
          }
          if (copyback_cursor_list[i].size() > 0) {
            copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                           copyback_cursor_list[i]);
          }
        }
        // Pointers in memcpy_address here will
        // be cast to ValuePtr<Tvalue>* in this funcation.
        ev->AllocateMemoryForNewFeatures(
            memcpy_address,
            init_cursor_list[0]);

        ev->SetDefaultValueOfNewFeatures(
            indices_host, N,
            init_cursor_list[0], memcpy_address,
            default_v, get_default_v_fn_,
            stream, event_mgr,
            ctx->eigen_gpu_device());

        ev->CopyEmbeddingsFromCPUToGPU(
            indices_host,
            copyback_cursor_list[0],
            memcpy_address,
            stream, event_mgr,
            ctx->eigen_gpu_device(),
            worker_threads);

        ev->CopyEmbeddingsToBuffer(
            out_base, N,
            slice_elems, memcpy_address,
            stream, event_mgr,
            ctx->eigen_gpu_device());
        delete []memcpy_address;

        if (ev->IsMultiLevel()) {
          ev->storage_manager()->Schedule([ev, indices_host, N]() {
            embedding::BatchCache<TFKey> *cache = ev->Cache();
            cache->add_to_rank(indices_host, N);
            delete[] indices_host;
          });
        }
      }

      Tensor *op_output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {batch_size, dimension},
                                               &op_output_tensor));
      auto op_output = op_output_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));

      Tensor *values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();

      this->lookuper_.set(
          i, out_base, op_output, values_offset, nnz,
          const_cast<int64_t *>(reinterpret_cast<const int64_t *>(sp_indices)));
    }

    if (this->combiner_ == "mean") {
      this->lookuper_.compute(ComputeEVFn<TKey, TValue, Mean>, batch_size,
                              this->dimension_, this->max_norm_,
                              device.stream());
    } else if (this->combiner_ == "sum") {
      this->lookuper_.compute(ComputeEVFn<TKey, TValue, Sum>, batch_size,
                              this->dimension_, this->max_norm_,
                              device.stream());
    } else {
      this->lookuper_.compute(ComputeEVFn<TKey, TValue, Sqrtn>, batch_size,
                              this->dimension_, this->max_norm_,
                              device.stream());
    }
  }

 private:
  std::map<uint64, int64> hash_map_;
  bool is_use_default_value_tensor_;
  std::function<TValue *(TValue *, TFKey, int64, int64, int64)>
      get_default_v_fn_;
  mutable easy_spinrwlock_t mu_ = EASY_SPINRWLOCK_INITIALIZER;
  bool *occupy_flag_ = nullptr;
  mutex m_init_occupy_flag_;
};


#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MultiKvResourceGather")                                  \
          .Device(DEVICE_GPU)                                        \
          .HostMemory("sp_indices")                                  \
          .HostMemory("dense_shape")                                 \
          .TypeConstraint<key_type_tf>("Tkeys")                      \
          .TypeConstraint<dtype>("dtype"),                           \
      MultiKvResourceGatherOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#undef REGISTER_GPU_KERNELS


template <typename TFKey, typename TKey, typename TValue>
class MultiEmbeddingSparseLookupOp
    : public GroupLookupForWardBaseOp<TKey, TValue> {
 public:
  explicit MultiEmbeddingSparseLookupOp(OpKernelConstruction *c)
      : GroupLookupForWardBaseOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext *ctx) override {
    int batch_size = -1;
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    for (int i = 0; i < this->num_lookups_; ++i) {
      const Tensor &emb_variable_tensor = ctx->input(i);
      const Tensor &sp_values_tensor = ctx->input(this->num_lookups_ + i);
      int64 emb_row_size = emb_variable_tensor.shape().dim_size(0);
      int64 emb_vec_size = emb_variable_tensor.shape().dim_size(1);

      const Tensor &sp_indices_tensor = ctx->input(this->num_lookups_ * 2 + i);
      int nnz = sp_indices_tensor.shape().dim_size(0);

      const Tensor &dense_shape_tensor = ctx->input(this->num_lookups_ * 3 + i);
      auto dense_shape = dense_shape_tensor.flat<int64>().data();

      if (batch_size == -1) {
        batch_size = dense_shape[0];
      } else {
        OP_REQUIRES(
            ctx, batch_size == dense_shape[0],
            errors::InvalidArgument(
                "shape[0] of each tensor in offset_indices are different."));
      }

      TensorShape emb_vectors_tensor_shape;

      emb_vectors_tensor_shape = TensorShape(std::vector<int64>(
          {static_cast<long long>(batch_size), emb_vec_size}));
      OP_REQUIRES(
          ctx, emb_vec_size == this->dimension_,
          errors::InvalidArgument(
              "shape[0] of each tensor in offset_indices are different."));
      Tensor *emb_vectors_tensor = nullptr;
      // allocate output
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &emb_vectors_tensor));
      auto emb_vectors = emb_vectors_tensor->flat<TValue>().data();

      // allocate offset tensor
      TensorShape values_offset_tensor_shape =
          TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));
      Tensor *values_offset_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(this->num_lookups_ + i,
                                               values_offset_tensor_shape,
                                               &values_offset_tensor));
      auto values_offset = values_offset_tensor->flat<int>().data();

      this->lookuper_.set(
          i,
          const_cast<TValue *>(reinterpret_cast<const TValue *>(
              emb_variable_tensor.flat<TValue>().data())),
          emb_vectors, values_offset, nnz,
          const_cast<int64_t *>(reinterpret_cast<const int64_t *>(
              sp_indices_tensor.flat<int64>().data())),
          const_cast<TKey *>(reinterpret_cast<const TKey *>(
              sp_values_tensor.flat<TFKey>().data())),
          emb_row_size);
    }
    if (this->combiner_ == "mean") {
      this->lookuper_.compute(ComputeSparseFn<TKey, TValue, Mean>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    } else if (this->combiner_ == "sum") {
      this->lookuper_.compute(ComputeSparseFn<TKey, TValue, Sum>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    } else {
      this->lookuper_.compute(ComputeSparseFn<TKey, TValue, Sqrtn>, batch_size,
                              this->dimension_, this->max_norm_, stream);
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type_tf, key_type, dtype_tf, dtype) \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MultiEmbeddingSparseLookUp")                             \
          .Device(DEVICE_GPU)                                        \
          .HostMemory("dense_shape")                                 \
          .TypeConstraint<key_type_tf>("Tkeys")                      \
          .TypeConstraint<dtype>("dtype"),                           \
      MultiEmbeddingSparseLookupOp<key_type_tf, key_type, dtype>)

REGISTER_GPU_KERNELS(int64, int64_t, float, float);
REGISTER_GPU_KERNELS(int32, int32_t, float, float);
#undef REGISTER_GPU_KERNELS

} //namespace tensorflow

#endif // GOOGLE_CUDA