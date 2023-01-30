#include <memory>

#include "Global/GlobalDefinitions.h"
#include "Presearcher/Presearcher.h"
#include "Tools/GpuPointer.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"

using log_utils::Logger;
using math_utils::round_up;
using std::make_unique;

namespace sbwt_search {

Presearcher::Presearcher(shared_ptr<GpuSbwtContainer> container_):
  container(container_) {}

auto Presearcher::presearch() -> void {
  constexpr const auto presearch_times
    = round_up<size_t>(1ULL << (presearch_letters * 2), threads_per_block);
  auto blocks_per_grid = presearch_times / threads_per_block;
  auto presearch_left = make_unique<GpuPointer<u64>>(presearch_times);
  auto presearch_right = make_unique<GpuPointer<u64>>(presearch_times);
  Logger::log_timed_event("PresearchFunction", Logger::EVENT_STATE::START);
  launch_presearch_kernel(presearch_left, presearch_right, blocks_per_grid);
  Logger::log_timed_event("PresearchFunction", Logger::EVENT_STATE::STOP);
  container->set_presearch(
    std::move(presearch_left), std::move(presearch_right)
  );
}

}  // namespace sbwt_search
