#pragma once

#include <mutex>

#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

namespace steam {

#if true
using StateKey = unsigned int;
using StateKeyHash = std::hash<unsigned int>;
inline StateKey NewStateKey() {
  static std::mutex mtx;
  static unsigned int id = 0;
  std::lock_guard<std::mutex> lock(mtx);
  return id++;
}
#else
/// alternatively we may use uuid so that there's no need for a mutex
using StateKey = boost::uuids::uuid;
using StateKeyHash = boost::hash<boost::uuids::uuid>;
inline StateKey NewStateKey() { return boost::uuids::random_generator()(); }
#endif

}  // namespace steam