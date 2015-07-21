#include "catch.hpp"

#include <steam/common/Time.hpp>

bool doubleCompare(double a, double b, double tol = 1e-6)
{
    return std::abs(a - b) <= tol;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// Sample Test
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("time test 1", "[time]" ) {

  steam::Time time;
  REQUIRE( time.nanosecs() == 0 );

  // example nanoseconds since epoch
  boost::int64_t epoch64 = 1500000000123456789;
  //                     ^   sec  ^^ nano  ^

  // test epoch
  steam::Time epoch = steam::Time(epoch64);
  REQUIRE( epoch.nanosecs() == 1500000000123456789 );

  // test epoch
  boost::int64_t epoch_secs64 = 1500000000e9;
  steam::Time epoch_secs = steam::Time(epoch_secs64);
  steam::Time epoch_nsecs = epoch - epoch_secs;
  REQUIRE( epoch_nsecs.nanosecs() == 123456789 );

  // test double
  double epoch_nsecs_float = epoch_nsecs.seconds();
  REQUIRE( doubleCompare(epoch_nsecs_float, 123456789e-9) );

  // double back to Time
  steam::Time nano_back_to_time(epoch_nsecs_float);
  REQUIRE( nano_back_to_time.nanosecs() == 123456789 );

  // two 32-bits to 64-bit (e.g. ros::Time)
  boost::int32_t secs32 = 1500000000;
  boost::int32_t nsecs32 = 123456789;
  REQUIRE( epoch.nanosecs() == steam::Time(secs32, nsecs32).nanosecs() );

  // doubles store 15 sig figs...
  //std::cout << std::setprecision(15) << std::fixed << epoch_nsecs.seconds() << std::resetiosflags(std::ios::fixed) << std::endl;
  //std::cout << std::setprecision(15) << std::fixed << epoch.seconds() << std::resetiosflags(std::ios::fixed) << std::endl;

} // TEST_CASE
