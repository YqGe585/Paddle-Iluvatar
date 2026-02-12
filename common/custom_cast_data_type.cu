// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "common/custom_device_func.h"
#include "paddle/common/hostdevice.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "thrust/device_ptr.h"

namespace phi {

// Mirror the cast functor behavior used in data_type_transform.cc.
template <typename InType, typename OutType>
struct CastDataTypeFunctor {
  HOSTDEVICE inline OutType operator()(InType in) const {
    return static_cast<OutType>(in);
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float8_e5m2,
                           phi::dtype::complex<float>> {
  HOSTDEVICE phi::dtype::complex<float> operator()(
      phi::dtype::float8_e5m2 in) const {
    return phi::dtype::complex<float>(static_cast<float>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float8_e5m2,
                           phi::dtype::complex<double>> {
  HOSTDEVICE phi::dtype::complex<double> operator()(
      phi::dtype::float8_e5m2 in) const {
    return phi::dtype::complex<double>(static_cast<double>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float8_e4m3fn,
                           phi::dtype::complex<float>> {
  HOSTDEVICE phi::dtype::complex<float> operator()(
      phi::dtype::float8_e4m3fn in) const {
    return phi::dtype::complex<float>(static_cast<float>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float8_e4m3fn,
                           phi::dtype::complex<double>> {
  HOSTDEVICE phi::dtype::complex<double> operator()(
      phi::dtype::float8_e4m3fn in) const {
    return phi::dtype::complex<double>(static_cast<double>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float16, phi::dtype::complex<float>> {
  HOSTDEVICE phi::dtype::complex<float> operator()(
      phi::dtype::float16 in) const {
    return phi::dtype::complex<float>(static_cast<float>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::float16, phi::dtype::complex<double>> {
  HOSTDEVICE phi::dtype::complex<double> operator()(
      phi::dtype::float16 in) const {
    return phi::dtype::complex<double>(static_cast<double>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::bfloat16, phi::dtype::complex<float>> {
  HOSTDEVICE phi::dtype::complex<float> operator()(
      phi::dtype::bfloat16 in) const {
    return phi::dtype::complex<float>(static_cast<float>(in));
  }
};

template <>
struct CastDataTypeFunctor<phi::dtype::bfloat16, phi::dtype::complex<double>> {
  HOSTDEVICE phi::dtype::complex<double> operator()(
      phi::dtype::bfloat16 in) const {
    return phi::dtype::complex<double>(static_cast<double>(in));
  }
};

// PointerToThrustDevicePtr has two specializations, one casts a (CUDA
// device) pointer into thrust::device_ptr, the other keeps rest types
// un-casted.
template <typename T, bool is_ptr>
struct PointerToThrustDevicePtr;

template <typename T>
struct PointerToThrustDevicePtr<T, true> {
  using ELEM = typename std::remove_pointer<T>::type;
  using RTYPE = thrust::device_ptr<ELEM>;

  inline thrust::device_ptr<ELEM> operator()(ELEM *ele) const {
    return thrust::device_pointer_cast(ele);
  }
};

template <typename T>
struct PointerToThrustDevicePtr<T, false> {
  using RTYPE = T;
  inline RTYPE operator()(RTYPE it) const { return it; }
};

// CastToCUDATransformIterator casts a pointer to thrust::device_ptr
// so it could be used as the iterator of thrust::transform.
template <typename T>
auto CastToCUDATransformIterator(T t) ->
    typename PointerToThrustDevicePtr<T, std::is_pointer<T>::value>::RTYPE {
  PointerToThrustDevicePtr<T, std::is_pointer<T>::value> cast;
  return cast(t);
}

// Implementation of CastImpl (free function)
template <typename InType, typename OutType>
void CastImpl(const CustomContext &dev_ctx,
              const void *in,
              void *out,
              int64_t numel) {
  auto *in_ptr = static_cast<const InType *>(in);
  auto *out_ptr = static_cast<OutType *>(out);
  auto *in_end = in_ptr + numel;

  auto stream = reinterpret_cast<cudaStream_t>(dev_ctx.stream());
  thrust::transform(thrust::cuda::par.on(stream),
                    CastToCUDATransformIterator(in_ptr),
                    CastToCUDATransformIterator(in_end),
                    CastToCUDATransformIterator(out_ptr),
                    CastDataTypeFunctor<InType, OutType>());
}

// Implementation of DispatchOut (free function)
template <typename InType>
void DispatchOut(const CustomContext &dev_ctx,
                 const void *in,
                 void *out,
                 int64_t numel,
                 DataType out_dtype) {
  switch (out_dtype) {
    case DataType::BOOL:
      CastImpl<InType, bool>(dev_ctx, in, out, numel);
      break;
    case DataType::UINT8:
      CastImpl<InType, uint8_t>(dev_ctx, in, out, numel);
      break;
    case DataType::INT16:
      CastImpl<InType, int16_t>(dev_ctx, in, out, numel);
      break;
    case DataType::INT32:
      CastImpl<InType, int32_t>(dev_ctx, in, out, numel);
      break;
    case DataType::INT64:
      CastImpl<InType, int64_t>(dev_ctx, in, out, numel);
      break;
    case DataType::FLOAT16:
      CastImpl<InType, phi::float16>(dev_ctx, in, out, numel);
      break;
    case DataType::BFLOAT16:
      CastImpl<InType, phi::bfloat16>(dev_ctx, in, out, numel);
      break;
    case DataType::FLOAT8_E4M3FN:
      CastImpl<InType, phi::float8_e4m3fn>(dev_ctx, in, out, numel);
      break;
    case DataType::FLOAT8_E5M2:
      CastImpl<InType, phi::float8_e5m2>(dev_ctx, in, out, numel);
      break;
    case DataType::FLOAT32:
      CastImpl<InType, float>(dev_ctx, in, out, numel);
      break;
    case DataType::FLOAT64:
      CastImpl<InType, double>(dev_ctx, in, out, numel);
      break;
    case DataType::COMPLEX64:
      CastImpl<InType, phi::complex64>(dev_ctx, in, out, numel);
      break;
    case DataType::COMPLEX128:
      CastImpl<InType, phi::complex128>(dev_ctx, in, out, numel);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Custom Transform does not support dst dtype %d.",
          static_cast<int>(out_dtype)));
  }
}

// Implementation of CustomDeviceFunc::CustomCastDataType
void CustomDeviceFunc::CustomCastDataType(const CustomContext &dev_ctx,
                                          const void *in,
                                          void *out,
                                          int64_t numel,
                                          DataType in_dtype,
                                          DataType out_dtype) const {
  switch (in_dtype) {
    case DataType::BOOL:
      DispatchOut<bool>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::UINT8:
      DispatchOut<uint8_t>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::INT16:
      DispatchOut<int16_t>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::INT32:
      DispatchOut<int32_t>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::INT64:
      DispatchOut<int64_t>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::FLOAT16:
      DispatchOut<phi::float16>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::BFLOAT16:
      DispatchOut<phi::bfloat16>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::FLOAT8_E4M3FN:
      DispatchOut<phi::float8_e4m3fn>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::FLOAT8_E5M2:
      DispatchOut<phi::float8_e5m2>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::FLOAT32:
      DispatchOut<float>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::FLOAT64:
      DispatchOut<double>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::COMPLEX64:
      DispatchOut<phi::complex64>(dev_ctx, in, out, numel, out_dtype);
      break;
    case DataType::COMPLEX128:
      DispatchOut<phi::complex128>(dev_ctx, in, out, numel, out_dtype);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Custom Transform does not support src dtype %d.",
          static_cast<int>(in_dtype)));
  }
}

}  // namespace phi
