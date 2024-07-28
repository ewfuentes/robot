#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace robot::sqlite3 {

class Database;

class Database {
   public:
    class Row;

    Database(const std::filesystem::path &path);
    ~Database();

    std::vector<Database::Row> query(const std::string &statement);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

   public:
    using Value =
        std::variant<std::nullopt_t, int, double, std::string, std::vector<unsigned char>>;

    class Row {
       public:
        Row(std::vector<Value> values, std::shared_ptr<std::vector<std::string>> column_names)
            : values_(std::move(values)), column_names_(std::move(column_names)) {}
        ~Row();
        const std::string &column_name(const int idx) const;
        const Value &value(const int idx) const;
        int size() const;

       private:
        std::vector<Value> values_;
        std::shared_ptr<std::vector<std::string>> column_names_;
    };
};

}  // namespace robot::sqlite3
