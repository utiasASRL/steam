#include "catch.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
/// Sample Test
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("example of a test case", "[example]" ) {

    int a = 5;
    REQUIRE( a == 5 );

    SECTION("if we change a to 10" ) {
        a = 10;
        REQUIRE( a == 10 );
    }

    SECTION("in a later section it is still 5" ) {
        REQUIRE( a == 5 );
    }
} // TEST_CASE
