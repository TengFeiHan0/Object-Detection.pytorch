#include <cuda_runtime_api.h>

namespace det {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace det
