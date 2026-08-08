
"""S9-N0 · control: harness DEFAULT config (plain L2 teacher objective), run through the
now-DEF-1-fixed decision layer, on the identical frozen surface as S9-N1. This isolates
whether S9-N1's large score drop comes from the epsilon-band objective itself or from
something else, since no baseline has been scored through the fixed harness yet."""
import sys, json
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import run

if __name__ == '__main__':
    out = run('S9-N0', 'control_default_l2_post_DEF1_fix')
    print(json.dumps(out, indent=1, default=str))
