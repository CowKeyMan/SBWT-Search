// Function credits:
// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g

#include "Tools/TypeDefinitions.h"

namespace memory_utils {

#ifdef __linux__

#include <unistd.h>
auto get_total_system_memory() -> u64 {
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

#elif _WIN32

#include <windows.h>
unsigned long long get_total_system_memory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}

#endif

}  // namespace memory_utils
