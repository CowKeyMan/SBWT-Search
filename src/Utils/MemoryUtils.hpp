
// Function credits:
// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g

namespace memory_utils {

#ifdef __linux__

#include <unistd.h>
unsigned long long get_total_system_memory() {
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
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
