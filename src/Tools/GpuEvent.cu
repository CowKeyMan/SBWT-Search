#include "Tools/GpuEvent.h"
#include "hip/hip_runtime.h"

namespace gpu_utils {

GpuEvent::GpuEvent(): element(reinterpret_cast<void *>(new hipEvent_t{})) {
  hipEventCreate(static_cast<hipEvent_t *>(element));
}

GpuEvent::~GpuEvent() { hipEventDestroy(*static_cast<hipEvent_t *>(element)); }

auto GpuEvent::get() const -> void * { return element; }

}  // namespace gpu_utils
