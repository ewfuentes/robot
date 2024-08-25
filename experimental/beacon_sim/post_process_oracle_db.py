
import argparse
from pathlib import Path
import sqlite3
import json

import experimental.beacon_sim.work_server_service_pb2 as wssp

def create_derivative_tables(con):
    con.execute('DROP TABLE IF EXISTS experiment_names;')
    con.execute('DROP TABLE IF EXISTS oracle_results;')
    con.execute('DROP INDEX IF EXISTS id_from_exp_name;')
    con.execute('''CREATE TABLE experiment_names
                (rowid INTEGER PRIMARY KEY ASC,
                 exp_name TEXT NOT NULL UNIQUE) STRICT;''')
    con.execute('''CREATE TABLE oracle_results
        (rowid INTEGER PRIMARY KEY ASC,
        exp_type INTEGER NOT NULL,
        trial_id INTEGER NOT NULL,
        eval_id INTEGER NOT NULL,
        nodes TEXT NOT NULL,
        expected_determinant REAL NOT NULL,
        expected_position_determinant REAL NOT NULL,
        prob_mass_in_region REAL NOT NULL,
        FOREIGN KEY(exp_type) REFERENCES experiment_names(rowid)
        ) STRICT;''')
    con.commit()

def populate_experiment_names(con):
    cur = con.execute("""
        SELECT job_inputs FROM job_table
        """)

    exp_names = set()
    for row in cur:
        msg = wssp.JobInputs()
        msg.ParseFromString(row[0])
        exp_names.add((Path(msg.results_file).stem,))

    con.executemany("INSERT INTO experiment_names (exp_name) VALUES (?);", list(exp_names))
    con.execute("CREATE UNIQUE INDEX id_from_exp_name ON experiment_names (exp_name);")
    con.commit()


def populate_oracle_results(con):

    cur = con.execute('SELECT job_inputs, job_result FROM job_table')
    insert_cursor = con.cursor()
    for (inputs, result) in cur:
        msg = wssp.JobInputs()
        msg.ParseFromString(inputs)
        exp_name = Path(msg.results_file).stem

        msg = wssp.JobResult()
        msg.ParseFromString(result)

        exp_id = con.execute('SELECT rowid FROM experiment_names WHERE exp_name=?', (exp_name,)).fetchone()[0]

        rows = []
        for plan in msg.plan:
            rows.append(
                (exp_id, plan.trial_id, plan.eval_trial_id, json.dumps(list(plan.nodes)),
                 plan.expected_determinant,
                 plan.expected_position_determinant,
                 plan.prob_mass_in_region
                 ))
        insert_cursor.executemany(""" INSERT INTO oracle_results
                        (exp_type, trial_id, eval_id, nodes,
                         expected_determinant,
                         expected_position_determinant,
                         prob_mass_in_region)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
        print(exp_name, exp_id)

    con.commit()


def main(db_path: Path):
    print(db_path)

    con = sqlite3.connect(db_path)

    # (re)create derivative tables
    create_derivative_tables(con)

    # populate experiment_names table
    populate_experiment_names(con)

    # populate the oracle_results table
    populate_oracle_results(con)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Post process results of oracle database')
    parser.add_argument('--db_path', required=True)
    
    args = parser.parse_args()

    main(Path(args.db_path))
