#pragma once

#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

namespace steam {

using StateKey = boost::uuids::uuid;
using StateKeyHash = boost::hash<boost::uuids::uuid>;
inline StateKey NewStateKey() { return boost::uuids::random_generator()(); }

}  // namespace steam