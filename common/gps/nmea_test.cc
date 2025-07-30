#include "gtest/gtest.h"
#include "nmea/message/gga.hpp"
#include "nmea/sentence.hpp"

namespace robot::gps {
TEST(NmeaTest, parse) {
    // Read an NMEA string from your serial port
    std::string nmea_string =
        "$GPGGA,172814.0,3723.46587704,N,12202.26957864,W,2,6,1.2,18.893,M,-25.669,M,2.0,0031*"
        "4F\r\n";

    // Parse the NMEA string into an NMEA sentence.
    nmea::sentence nmea_sentence(nmea_string);

    EXPECT_EQ(nmea_sentence.type(), "GGA");

    // Parse GGA message from NMEA sentence.
    nmea::gga gga(nmea_sentence);

    EXPECT_TRUE(gga.utc.exists());
    EXPECT_DOUBLE_EQ(gga.utc.get(), 62894);

    EXPECT_TRUE(gga.latitude.exists());
    EXPECT_NEAR(gga.latitude.get(), 37.3911, 1e-5);

    EXPECT_TRUE(gga.longitude.exists());
    EXPECT_NEAR(gga.longitude.get(), -122.037826, 1e-5);
}

TEST(NmeaTest, generate) {
    // Create NMEA sentence to populate.
    // For this custom/proprietary message, the talker is XY and the type is CMD.
    // The message has 3 fields.
    nmea::sentence nmea_sentence("XY", "CMD", 3);

    // Populate the first two fields using their field index.
    // The third field is optional and left blank.
    nmea_sentence.set_field(0, std::to_string(1.23));
    nmea_sentence.set_field(1, "abc");

    // Test
    EXPECT_EQ(3, nmea_sentence.field_count());
    EXPECT_EQ(std::stod(nmea_sentence.get_field(0)), 1.23);
    EXPECT_EQ(nmea_sentence.get_field(1), "abc");
    EXPECT_EQ(nmea_sentence.get_field(2), "");
}
}  // namespace robot::gps