
#include "common/sqlite3/sqlite3.hh"

#include <concepts>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <variant>

#include "gtest/gtest.h"

namespace robot::sqlite3 {

TEST(Sqlite3Test, open_database_test) {
    // Setup
    const char *tmp_dir = std::getenv("TEST_TMPDIR");
    const std::filesystem::path db_path = std::filesystem::path(tmp_dir) / "test_query.db";

    Database db(db_path);
    // Action
    db.query(
        "CREATE TABLE test_table (rowid INTEGER PRIMARY KEY ASC, name TEXT NOT NULL, age REAL) "
        "STRICT;");
    db.query("INSERT INTO test_table (rowid,name,age) VALUES (123,'hello',789.456);");
    const auto rows = db.query("SELECT * FROM test_table;");

    // Verification
    for (const auto &row : rows) {
        for (int i = 0; i < row.size(); i++) {
            std::visit(
                [](const auto &arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::nullopt_t>) {
                    } else if constexpr (std::is_same_v<T, std::vector<unsigned char>>) {
                    } else if constexpr (std::is_same_v<T, int>) {
                        EXPECT_EQ(arg, 123);
                    } else if constexpr (std::is_same_v<T, double>) {
                        EXPECT_NEAR(arg, 789.456, 1e-6);
                    } else {
                        EXPECT_EQ(arg, "hello");
                    }
                },
                row.value(i));
        }
    }
}

TEST(Sqlite3Test, bind_statement_test) {
    // Setup
    const char *tmp_dir = std::getenv("TEST_TMPDIR");
    const std::filesystem::path db_path = std::filesystem::path(tmp_dir) / "test_bind_statement.db";
    Database db(db_path);

    // Action
    db.query(
        "CREATE TABLE test_table (rowid INTEGER PRIMARY KEY ASC, name TEXT NOT NULL, age REAL) "
        "STRICT;");
    const auto statement =
        db.prepare("INSERT INTO test_table (rowid,name,age) VALUES (:rowid,:name,:age);");
    db.bind(statement, {{":rowid", 10}, {":name", "Foo"}, {":age", 34.456}});
    db.step(statement);

    const auto rows = db.query("SELECT * FROM test_table;");

    // Verification
    for (const auto &row : rows) {
        for (int i = 0; i < row.size(); i++) {
            std::visit(
                [](const auto &arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::nullopt_t>) {
                    } else if constexpr (std::is_same_v<T, std::vector<unsigned char>>) {
                    } else if constexpr (std::is_same_v<T, int>) {
                        EXPECT_EQ(arg, 10);
                    } else if constexpr (std::is_same_v<T, double>) {
                        EXPECT_NEAR(arg, 34.456, 1e-6);
                    } else {
                        EXPECT_EQ(arg, "Foo");
                    }
                },
                row.value(i));
        }
    }
}
}  // namespace robot::sqlite3
