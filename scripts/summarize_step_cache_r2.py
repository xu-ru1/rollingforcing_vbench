#!/usr/bin/env python3
import argparse, json, math, os
from collections import Counter


def read_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--decisions', required=True)
    ap.add_argument('--executions', required=True)
    ap.add_argument('--runtime-summary', required=True)
    ap.add_argument('--expected-reuse', type=int, required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    decisions=read_jsonl(args.decisions)
    executions=read_jsonl(args.executions)
    runtime=json.load(open(args.runtime_summary, encoding='utf-8'))
    errors=[]
    stage_counts=Counter(int(r['local_stage_index']) for r in decisions)
    execute=[r for r in decisions if r.get('execute_reuse')]
    if len(decisions)!=35: errors.append(f'decision_count={len(decisions)} expected=35')
    if dict(sorted(stage_counts.items()))!={0:7,1:7,2:7,3:7,4:7}: errors.append(f'stage_counts={dict(stage_counts)}')
    if len(execute)!=args.expected_reuse: errors.append(f'execute_reuse={len(execute)} expected={args.expected_reuse}')
    if any(not r.get('all_required_residuals_valid') for r in execute): errors.append('reuse_with_missing_residual')
    if len(executions)!=330: errors.append(f'layer_execution_count={len(executions)} expected=330')
    totals=runtime['operator_totals']
    if totals['k_tokens']!=totals['v_tokens'] or totals['k_tokens']!=totals['full_window_tokens']:
        errors.append('K/V tokens do not equal full-window tokens')
    expected_layer_reuse=args.expected_reuse*30
    if totals['reuse_block_events']!=expected_layer_reuse:
        errors.append(f"reuse_block_events={totals['reuse_block_events']} expected={expected_layer_reuse}")
    if args.expected_reuse==0:
        if totals['q_tokens']!=totals['k_tokens']: errors.append('all-recompute q_tokens != k_tokens')
    else:
        if not totals['q_tokens']<totals['k_tokens']: errors.append('sparse case did not reduce q tokens')
        for name in ('attention_output_tokens','cross_attention_tokens','mlp_tokens'):
            if totals[name]!=totals['q_tokens']: errors.append(f'{name} != q_tokens')
    report={
        'status':'ok' if not errors else 'error',
        'decision_count':len(decisions),
        'stage_counts':dict(sorted(stage_counts.items())),
        'execute_reuse_count':len(execute),
        'expected_execute_reuse_count':args.expected_reuse,
        'layer_execution_count':len(executions),
        'operator_totals':totals,
        'peak_residual_bytes':runtime.get('peak_residual_bytes'),
        'reuse_points':[[r['global_video_block_id'],r['local_stage_index']] for r in execute],
        'errors':errors,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(report,sort_keys=True))
    if errors: raise SystemExit(2)

if __name__=='__main__': main()
