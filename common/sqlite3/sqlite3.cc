
#include "common/sqlite3/sqlite3.hh"

#include "common/check.hh"
#include "sqlite3/sqlite3.h"

namespace robot::sqlite3 {
struct Database::Impl {
    Impl(const std::filesystem::path &path) { check_result(sqlite3_open(path.c_str(), &db_)); }
    ~Impl() { check_result(sqlite3_close(db_)); }
    struct sqlite3 *db() { return db_; }

    void check_result(const int error_code) {
        CHECK(error_code == SQLITE_OK, sqlite3_errmsg(db_));
    };

    std::vector<Row> query(const std::string &statement) {
        struct OutParams {
            std::vector<Row> rows;
            std::shared_ptr<std::vector<std::string>> column_names;
        };
        OutParams out;

        sqlite3_stmt *stmt;
        check_result(sqlite3_prepare_v2(db_, statement.c_str(), statement.size(), &stmt, nullptr));

        int result = SQLITE_OK;
        const int num_columns = sqlite3_column_count(stmt);
        while ((result = sqlite3_step(stmt)) != SQLITE_DONE) {
            CHECK(result == SQLITE_ROW);

            if (out.column_names == nullptr) {
                out.column_names = std::make_shared<std::vector<std::string>>();
                for (int i = 0; i < num_columns; i++) {
                    out.column_names->push_back(sqlite3_column_name(stmt, i));
                }
            }

            std::vector<Database::Value> values;
            for (int i = 0; i < num_columns; i++) {
                switch (sqlite3_column_type(stmt, i)) {
                    case SQLITE_NULL: {
                        values.push_back(std::nullopt);
                        break;
                    }
                    case SQLITE_INTEGER: {
                        values.push_back(sqlite3_column_int(stmt, i));
                        break;
                    }
                    case SQLITE_FLOAT: {
                        values.push_back(sqlite3_column_double(stmt, i));
                        break;
                    }
                    case SQLITE3_TEXT: {
                        values.push_back(
                            reinterpret_cast<const char *>(sqlite3_column_text(stmt, i)));
                        break;
                    }
                    case SQLITE_BLOB: {
                        const int size = sqlite3_column_bytes(stmt, i);
                        const unsigned char *buf =
                            reinterpret_cast<const unsigned char *>(sqlite3_column_blob(stmt, i));
                        values.emplace_back(std::vector(buf, buf + size));
                    }
                }
            }
            out.rows.emplace_back(Row(std::move(values), out.column_names));
        }
        sqlite3_finalize(stmt);
        return out.rows;
    }
    struct sqlite3 *db_;
};

Database::Row::~Row() = default;

const std::string &Database::Row::column_name(const int idx) const {
    return column_names_->at(idx);
}

int Database::Row::size() const { return column_names_->size(); }

const Database::Value &Database::Row::value(const int idx) const { return values_.at(idx); }

Database::~Database() = default;
Database::Database(const std::filesystem::path &path)
    : impl_(std::make_unique<Database::Impl>(path)) {}

std::vector<Database::Row> Database::query(const std::string &statement) {
    return impl_->query(statement);
}
}  // namespace robot::sqlite3
