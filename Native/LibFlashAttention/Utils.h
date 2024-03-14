#pragma once

#include "torch/torch.h"
#include <string>

typedef torch::Tensor* Tensor;
typedef torch::Generator* Generator;

extern thread_local char* flash_last_err = nullptr;

#if _WIN32
#define strdup _strdup
#endif

#define CATCH(x) \
  try { \
    flash_last_err = 0; \
    x \
  } catch (const c10::Error& e) { \
      flash_last_err = strdup(e.what()); \
  } catch (const std::runtime_error e) { \
      flash_last_err = strdup(e.what()); \
  }

// Return undefined tensors as nullptr to C#
inline Tensor ResultTensor(const at::Tensor& res)
{
    if (res.defined())
        return new torch::Tensor(res);
    else
        return nullptr;
}