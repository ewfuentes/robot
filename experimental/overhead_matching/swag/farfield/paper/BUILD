py_library(
    name = "table_common",
    srcs = ["table_common.py"],
)

py_binary(
    name = "dataset_table",
    srcs = ["dataset_table.py"],
    deps = [":table_common"],
)

py_binary(
    name = "results_table",
    srcs = ["results_table.py"],
    deps = [":table_common"],
)

py_test(
    name = "tables_test",
    srcs = ["tables_test.py"],
    deps = [
        ":dataset_table",
        ":results_table",
        ":table_common",
    ],
)
