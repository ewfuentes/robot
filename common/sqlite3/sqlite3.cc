
#include "common/sqlite3/sqlite3.hh"

#include "common/check.hh"
#include "sqlite3/sqlite3.h"

namespace robot::sqlite3 {

struct Database::Statement::Impl {
    std::shared_ptr<sqlite3_stmt> stmt;
    std::shared_ptr<std::vector<std::string>> column_names;
};

struct Database::Impl {
    Impl(const std::filesystem::path &path) { check_result(sqlite3_open(path.c_str(), &db_), path); }
    ~Impl() { check_result(sqlite3_close(db_)); }
    struct sqlite3 *db() { return db_; }

    template <typename...Ts>
    void check_result(const int error_code, Ts... args) {
        CHECK(error_code == SQLITE_OK, sqlite3_errmsg(db_), args...);
    };

    Database::Statement prepare(const std::string &statement) {
        sqlite3_stmt *stmt;
        check_result(sqlite3_prepare_v2(db_, statement.c_str(), statement.size(), &stmt, nullptr), statement);
        return Statement{
            .impl_ = std::make_unique<Database::Statement::Impl>(Database::Statement::Impl{
                .stmt = std::shared_ptr<sqlite3_stmt>(
                    stmt, [this](sqlite3_stmt *stmt) { check_result(sqlite3_finalize(stmt)); }),
                .column_names = nullptr,
            })};
    }

    void bind(const Statement &stmt, const std::unordered_map<std::string, Database::Value> &args) {
        sqlite3_stmt *stmt_ptr = stmt.impl_->stmt.get();
        CHECK(args.size() == sqlite3_bind_parameter_count(stmt_ptr),
              "insufficient number of arguments", sqlite3_sql(stmt_ptr), args);
        for (const auto &[key, value] : args) {
            const int param_idx = sqlite3_bind_parameter_index(stmt_ptr, key.c_str());
            std::visit(
                [param_idx, stmt_ptr, this](const auto &arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::nullopt_t>) {
                        check_result(sqlite3_bind_null(stmt_ptr, param_idx));
                    } else if constexpr (std::is_same_v<T, int>) {
                        check_result(sqlite3_bind_int(stmt_ptr, param_idx, arg));
                    } else if constexpr (std::is_same_v<T, double>) {
                        check_result(sqlite3_bind_double(stmt_ptr, param_idx, arg));
                    } else if constexpr (std::is_same_v<T, std::vector<unsigned char>>) {
                        check_result(sqlite3_bind_blob(stmt_ptr, param_idx, arg.data(), arg.size(),
                                                       SQLITE_TRANSIENT));
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        constexpr int TO_FIRST_TERMINATOR = -1;
                        check_result(sqlite3_bind_text(stmt_ptr, param_idx, arg.c_str(),
                                                       TO_FIRST_TERMINATOR, SQLITE_TRANSIENT));
                    }
                },
                value);
        }
    }

    std::optional<Row> step(const Statement &stmt) {
        sqlite3_stmt *stmt_ptr = stmt.impl_->stmt.get();
        const int result = sqlite3_step(stmt_ptr);
        if (result == SQLITE_DONE) {
            return std::nullopt;
        }
        CHECK(result == SQLITE_ROW, sqlite3_errmsg(db_));
        const int num_columns = sqlite3_column_count(stmt_ptr);
        if (stmt.impl_->column_names == nullptr) {
            std::vector<std::string> column_names;
            for (int i = 0; i < num_columns; i++) {
                column_names.push_back(sqlite3_column_name(stmt_ptr, i));
            }
            stmt.impl_->column_names =
                std::make_shared<std::vector<std::string>>(std::move(column_names));
        }

        std::vector<Database::Value> values;
        for (int i = 0; i < num_columns; i++) {
            switch (sqlite3_column_type(stmt_ptr, i)) {
                case SQLITE_NULL: {
                    values.push_back(std::nullopt);
                    break;
                }
                case SQLITE_INTEGER: {
                    values.push_back(sqlite3_column_int(stmt_ptr, i));
                    break;
                }
                case SQLITE_FLOAT: {
                    values.push_back(sqlite3_column_double(stmt_ptr, i));
                    break;
                }
                case SQLITE3_TEXT: {
                    values.push_back(
                        reinterpret_cast<const char *>(sqlite3_column_text(stmt_ptr, i)));
                    break;
                }
                case SQLITE_BLOB: {
                    const int size = sqlite3_column_bytes(stmt_ptr, i);
                    const unsigned char *buf =
                        reinterpret_cast<const unsigned char *>(sqlite3_column_blob(stmt_ptr, i));
                    values.emplace_back(std::vector(buf, buf + size));
                }
            }
        }
        return Row(std::move(values), stmt.impl_->column_names);
    }

    void reset(const Statement &stmt) {
        sqlite3_stmt *stmt_ptr = stmt.impl_->stmt.get();
        sqlite3_reset(stmt_ptr);
    }

    std::vector<Row> query(const std::string &statement) {
        Statement stmt = prepare(statement);

        std::vector<Row> out;
        for (std::optional<Row> r = step(stmt); r.has_value(); r = step(stmt)) {
            out.push_back(r.value());
        }
        return out;
    }
    struct sqlite3 *db_;
};

Database::Row::~Row() = default;

const std::string &Database::Row::column_name(const int idx) const {
    return column_names_->at(idx);
}

int Database::Row::size() const { return column_names_->size(); }

const Database::Value &Database::Row::value(const int idx) const { return values_.at(idx); }

Database::Statement::~Statement() = default;

Database::~Database() = default;
Database::Database(const std::filesystem::path &path)
    : impl_(std::make_unique<Database::Impl>(path)) {}

std::vector<Database::Row> Database::query(const std::string &statement) {
    return impl_->query(statement);
}

Database::Statement Database::prepare(const std::string &statement) {
    return impl_->prepare(statement);
}

void Database::bind(const Database::Statement &stmt,
                    const std::unordered_map<std::string, Database::Value> &args) {
    impl_->bind(stmt, args);
}

std::optional<Database::Row> Database::step(const Database::Statement &stmt) {
    return impl_->step(stmt);
}

void Database::reset(const Database::Statement &stmt) { return impl_->reset(stmt); }

}  // namespace robot::sqlite3
