#pragma once

#include <cstdint>
#include <functional>

namespace steam {
namespace traj {

/** \brief STEAM trajectory time class */
class Time {
 public:
  Time() : nsecs_(0) {}
  Time(int64_t nsecs) : nsecs_(nsecs) {}
  Time(double secs) : nsecs_(secs * 1e9) {}
  Time(int32_t secs, int32_t nsec) {
    int64_t t1 = (int64_t)secs;
    int64_t t2 = (int64_t)nsec;
    this->nsecs_ = t1 * 1000000000 + t2;
  }

  /**
   * \brief Get number of seconds in double format.
   * \note depending on scale of time, this may cause a loss in precision (for
   * example, if nanosecond since epoch are stored). Usually it makes sense to
   * use this method after *this* has been reduced to a duration between two
   * times.
   */
  double seconds() const { return static_cast<double>(nsecs_) * 1e-9; }

  /** \brief Get number of nanoseconds in int format. */
  const int64_t& nanosecs() const { return nsecs_; }

  /// addition/subtraction operators
  Time& operator+=(const Time& other) {
    nsecs_ += other.nsecs_;
    return *this;
  }

  Time operator+(const Time& other) const {
    Time temp(*this);
    temp += other;
    return temp;
  }

  Time& operator-=(const Time& other) {
    nsecs_ -= other.nsecs_;
    return *this;
  }

  Time operator-(const Time& other) const {
    Time temp(*this);
    temp -= other;
    return temp;
  }

  /// comparison operators
  bool operator==(const Time& other) const { return nsecs_ == other.nsecs_; }
  bool operator!=(const Time& other) const { return nsecs_ != other.nsecs_; }
  bool operator<(const Time& other) const { return nsecs_ < other.nsecs_; }
  bool operator>(const Time& other) const { return nsecs_ > other.nsecs_; }
  bool operator<=(const Time& other) const { return nsecs_ <= other.nsecs_; }
  bool operator>=(const Time& other) const { return nsecs_ >= other.nsecs_; }

 private:
  /**
   * \brief int64 gives nanosecond precision since epoch (+/- ~9.2e18), which
   * covers the ~1.5e9 seconds since epoch and 1e9 nsecs. Furthermore, a single
   * base type, rather than two combined unsigned int32s to allow nsecs to be
   * used as a key in a std::map.
   */
  int64_t nsecs_;
};

}  // namespace traj
}  // namespace steam

// Specialization of std:hash function
namespace std {

template <>
struct hash<steam::traj::Time> {
  std::size_t operator()(const steam::traj::Time& k) const noexcept {
    return hash<int64_t>{}(k.nanosecs());
  }
};

}  // namespace std
