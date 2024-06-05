
#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "BS_thread_pool.hpp"


namespace robot {
TEST(ThreadPoolTest, future_test) {
    // Setup
    // Action + Verification
    
    BS::thread_pool pool(4);
    std::array<int, 2> results = {0, 0};
    pool.detach_task([result = &results[0]](){
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        *result = 1;
    });
    pool.detach_task([result = &results[1]](){
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        *result = 2;
    });
    pool.wait();

    std::cout << results[0] << " " << results[1] << std::endl;
}
}
