#include <ostream>
#include <string>

#include "IndexFileParser/IndexFileParser.hpp"
#include "IndexWriter/IndexWriter.hpp"
#include "Utils/BitVectorUtils.h"
#include "Utils/IOUtils.hpp"

using std::ofstream;
using std::string;

namespace sbwt_search {

auto SdslIndexWriter::do_write() const -> void {
  ThrowingOfstream stream(path, std::ios::out | std::ios::binary);
  size_t string_size = format.size();
  stream.write(reinterpret_cast<char *>(&string_size), sizeof(u64));
  stream << format;
  for (auto letter = 0; letter < 4; ++letter) {
    auto v = container.get_acgt_sdsl(static_cast<ACGT>(letter));
    v.serialize(stream);
  }
}

auto BitVectorIndexWriter::do_write() const -> void {
  for (auto i = 0; i < 4; ++i) {
    ThrowingOfstream stream(
      path + acgt_postfixes[i], std::ios::out | std::ios::binary
    );
    auto bits_total = container.get_bits_total();
    stream.write(reinterpret_cast<char *>(&bits_total), sizeof(u64));
    auto v = container.get_acgt(static_cast<ACGT>(i));
    stream.write(
      reinterpret_cast<char *>(const_cast<u64 *>(v)),
      container.get_bit_vector_size() * sizeof(u64)
    );
  }
}

}
