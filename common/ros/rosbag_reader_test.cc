
#include "gtest/gtest.h"

#include "embag/view.h"

namespace robot::ros {
GTEST_TEST(RosbagReaderTest, read_rosbag) {
    std::vector<std::string> rosbag_paths = {
        "/home/erick/scratch/symphony_lake_bags/20170307/2017-03-07-08-52-30_0.bag",
        "/home/erick/scratch/symphony_lake_bags/20170307/2017-03-07-08-53-20_1.bag",
    };
    Embag::View view;
    for (const auto &path : rosbag_paths) {
        view.addBag(path);
    }

    const auto topics = view.topics();
    std::cout << "Topics:" << std::endl;
    for (const auto &topic: topics) {
        std::cout << topic << std::endl;
    }
    view.getMessages();

    std::cout << "Messages:" << std::endl;
    for (const auto &message : view) {
        std::cout << message->topic << " " <<  message->getTypeName() << std::endl;
        std::cout << message->toString() << std::endl;
    }
}
}
