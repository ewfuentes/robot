#include "SQLiteCpp/SQLiteCpp.h"

#include <iostream>

#include "gtest/gtest.h"

namespace common::database {

TEST(sqlitecpp_test, in_memory_test) {
    // Use in-memory DB (lives only for the lifetime of 'db')
    SQLite::Database db(":memory:", SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

    // Create a table
    db.exec("CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY, name TEXT)");

    // Insert data
    SQLite::Statement insert(db, "INSERT INTO people (name) VALUES (?)");
    insert.bind(1, "Dog");
    insert.exec();

    // Query data
    SQLite::Statement query(db, "SELECT id, name FROM people");

    int rows_found = 0;
    while (query.executeStep()) {
        int id = query.getColumn(0);
        std::string name = query.getColumn(1);

        EXPECT_EQ(id, 1);
        EXPECT_EQ(name, "Dog");
        ++rows_found;
    }

    EXPECT_EQ(rows_found, 1);  // Ensure we actually got a row
}

}  // namespace common::database
