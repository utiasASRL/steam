#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// catch.hpp Usage Note
///
/// REQUIRE -- terminates on failure
/// CHECK   -- continues test on failure
/// INFO    -- buffers a string that is printed on failure (cleared after scope ends)
///
/// TEST_CASE -- a unit test
/// SECTION -- resets initialization in TEST_CASE for various tests
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// Sample Test
/////////////////////////////////////////////////////////////////////////////////////////////
//TEST_CASE("example of a test case", "[example]" ) {

//    int a = 5;
//    REQUIRE( a == 5 );

//    SECTION("if we change a to 10" ) {
//        a = 10;
//        REQUIRE( a == 10 );
//    }

//    SECTION("in a later section it is still 5" ) {
//        REQUIRE( a == 5 );
//    }
//} // TEST_CASE

