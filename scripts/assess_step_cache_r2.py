#!/usr/bin/env python3
import argparse, json, os

def load(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--output', required=True)
    a=ap.parse_args()
    root=a.run_root
    paths={
        'attention_oracle': os.path.join(root,'reports','attention_oracle.json'),
        'all_recompute': os.path.join(root,'reports','all_recompute_summary.json'),
        'few': os.path.join(root,'reports','few_summary.json'),
        'more': os.path.join(root,'reports','more_summary.json'),
        'all_recompute_video': os.path.join(root,'reports','all_recompute_video_check.json'),
        'few_video': os.path.join(root,'reports','few_video_check.json'),
        'more_video': os.path.join(root,'reports','more_video_check.json'),
    }
    errors=[]; reports={}
    for name,path in paths.items():
        if not os.path.isfile(path):
            errors.append(f'missing:{name}:{path}'); continue
        reports[name]=load(path)
        if reports[name].get('status') not in ('ok',):
            errors.append(f'failed:{name}')
    if 'few' in reports and 'more' in reports:
        f=reports['few']['operator_totals']; m=reports['more']['operator_totals']
        if not m['q_tokens'] < f['q_tokens']:
            errors.append('more_injection_did_not_reduce_q_tokens_below_few')
        if not m['reuse_block_events'] > f['reuse_block_events']:
            errors.append('more_injection_reuse_count_not_above_few')
    payload={'status':'ok' if not errors else 'error','errors':errors,'reports':paths}
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    with open(a.output,'w',encoding='utf-8') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(payload,sort_keys=True))
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
