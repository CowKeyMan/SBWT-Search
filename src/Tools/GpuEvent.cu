#include "Tools/GpuEvent.h"
#include "Tools/GpuUtils.h"
#include "hip/hip_runtime.h"

namespace gpu_utils {

GpuEvent::GpuEvent(): element(static_cast<void *>(new hipEvent_t{})) {
  hipEventCreate(static_cast<hipEvent_t *>(element));
}

GpuEvent::~GpuEvent() { hipEventDestroy(*static_cast<hipEvent_t *>(element)); }

auto GpuEvent::record(GpuStream *s) -> void {
  GPU_CHECK(hipEventRecord(
    *static_cast<hipEvent_t *>(element),
    s == nullptr ? nullptr : *static_cast<hipStream_t *>(s->get())
  ));
}

auto GpuEvent::get() const -> void * { return element; }

auto GpuEvent::time_elapsed_ms(const GpuEvent &e) -> float {
  float millis = -1;
  GPU_CHECK(hipEventElapsedTime(
    &millis,
    *static_cast<hipEvent_t *>(element),
    *static_cast<hipEvent_t *>(e.get())
  ));
  return millis;
}

}  // namespace gpu_utils
