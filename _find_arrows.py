import re, sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

files = ['app.py', 'monitor_agent.py', 'risk_agent.py', 'reco_agent.py']

print("=== Direction / arrow patterns ===")
for f in files:
    src = open(f, encoding='utf-8', errors='replace').read()
    lines = src.splitlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if any(x in s for x in [
            'direction_f', 'increased', 'decreased',
            'direction_icon', 'score_change',
            'direction_ctx', 'worsened', 'improved',
            'changed_fields', 'pct_change',
        ]):
            # Replace non-ascii for safe print
            safe = s.encode('ascii', 'replace').decode('ascii')[:120]
            print(f"  {f}:{i}: {safe}")
