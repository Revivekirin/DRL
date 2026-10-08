"""분석 도구 전용 unittest. simulator/learner/pytest conftest를 실행하지 않는다."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

spec=importlib.util.spec_from_file_location('analysis_tool',Path(__file__).parents[1]/'scripts/analyze_runs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class AnalysisTests(unittest.TestCase):
    def test_stream_and_flat(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';p.write_text('{"a": [1, 2]}\n{"a": [3]}\n')
            self.assertEqual([dict(module.flat(r)) for r in module.stream(p)],[{'a/0':1,'a/1':2},{'a/0':3}])
            p.write_text('{bad}\n')
            with self.assertRaisesRegex(ValueError,':1:'):list(module.stream(p))
    def test_quantiles_finite_extremes_and_signed_q(self):
        result=module.stats([-1000,-2,0,1,1e9,float('nan')])
        self.assertEqual(result['nonfinite'],1);self.assertEqual(result['median'],0)
        self.assertEqual(result['minimum'],-1000);self.assertEqual(result['maximum'],1e9)
    def test_generation_boundary(self):
        g=[{'refit':1,'real_env_steps':4000},{'refit':2,'real_env_steps':6016}]
        self.assertIsNone(module.generation_for(3999,g))
        self.assertEqual(module.generation_for(6015,g),1)
        self.assertEqual(module.generation_for(6016,g),2)
    def test_csv_union_preserves_events(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.csv';module.csvfile(p,[{'step':32,'event':1},{'step':32,'event':2,'value':9}])
            self.assertEqual(len(p.read_text().splitlines()),3)
            with self.assertRaises(FileExistsError):module.csvfile(p,[])

    def test_streaming_audit_and_corrupt_event(self):
        import sys
        sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
        from dynamics_shift.utils.tracking_audit import audit
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'metrics').mkdir()
            counts=dict(real_env_steps=1,vector_steps=1,policy_gradient_steps=1,episodes=0,
                        dynamics_model_refit_count=0,synthetic_transition_count=0,
                        real_policy_samples=1,synthetic_policy_samples=0)
            (p/'metadata.json').write_text(json.dumps(dict(status='complete',**counts)))
            (p/'tracking_summary.json').write_text('{"tracking_errors":0}')
            local=dict(counts,wall_time_seconds=1,model_replay_size=0)
            (p/'metrics/train.jsonl').write_text(json.dumps(local)+'\n')
            first={'metric_event_id':1,'real_env_steps':1,'policy_gradient_steps':1,
                   'policy_update/actor_loss':0,'policy_update/policy_gradient_steps':1,
                   'policy_update/real_env_steps':1,'policy_update/batch_real_count':1,
                   'policy_update/batch_synthetic_count':0,'policy_update/actual_synthetic_ratio':0}
            second={'metric_event_id':2,'real_env_steps':1,**{'train/'+k:v for k,v in local.items()}}
            rows=[dict(metrics=m,delivery='offline_local') for m in (first,second)]
            path=p/'tracking_metrics.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
            self.assertEqual(audit(p)['status'],'PASS')
            rows[1]['metrics']['metric_event_id']=3
            path.write_text('\n'.join(map(json.dumps,rows)))
            with self.assertRaises(AssertionError):audit(p)

if __name__=='__main__':unittest.main()
