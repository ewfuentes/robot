import unittest
import json,os,subprocess,tempfile
from pathlib import Path
from prepare_workspace import prepare,PACKAGE
from run_sequences import select_rows
class ShardingTest(unittest.TestCase):
    def test_all_thirteen_exactly_once(self):
        rows=[dict(dataset=f'd{i}') for i in range(13)]
        shards=[select_rows(rows,None,i,4) for i in range(4)]
        names=[r['dataset'] for s in shards for r in s]
        self.assertEqual(len(names),13)
        self.assertEqual(set(names),{r['dataset'] for r in rows})
        self.assertEqual(len(names),len(set(names)))
        with self.assertRaises(ValueError):select_rows(rows,['missing'],0,4)
        with self.assertRaises(ValueError):select_rows(rows,None,4,4)
class LauncherTest(unittest.TestCase):
    def test_installed_launcher_preserves_selected_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';output=root/'output'
            rows=json.loads((PACKAGE/'overnight_all13_manifest.json').read_text())['rows']
            for row in rows:
                stem=f"{row['dataset']}.dedup.geometry_integrated_v1_prefix{row['n_frames']-2}.seed0"
                for rel in [f'results/{stem}.json',f'logs/{stem}.run.json',f"tables/{row['dataset']}.dedup.audit_only_v2.report.json"]:
                    path=source/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            prepare(source,output,root/'repo with spaces')
            interpreter=root/'fake-python';interpreter.write_text('#!/bin/sh\nprintf "%s" "$CUDA_VISIBLE_DEVICES"\n');interpreter.chmod(0o755)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',FARFIELD_PYTHON=str(interpreter))
            actual=subprocess.check_output([str(output/'tools/python.sh')],env=env,text=True)
            self.assertEqual(actual,'3')
            with self.assertRaises(ValueError):prepare(source,output,root/'repo')
if __name__=='__main__':unittest.main()
