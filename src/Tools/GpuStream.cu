#include "Tools/GpuStream.h"
#include "hip/hip_runtime.h"

namespace gpu_utils {

// NOLINTNEXTLINE (cppcoreguidelines-pro-type-reinterpret-cast)
GpuStream::GpuStream(): element(reinterpret_cast<void *>(new hipStream_t{})) {
  hipStreamCreate(static_cast<hipStream_t *>(element));
}

GpuStream::~GpuStream() {
  hipStreamDestroy(*static_cast<hipStream_t *>(element));
}

auto GpuStream::get() const -> void * { return element; }

}  // namespace gpu_utils
