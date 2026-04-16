import re, sys
sys.stdout.reconfigure(encoding='utf-8')
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines()
for i, line in enumerate(lines, 1):
    if any(k in line for k in ['save_health', 'run_data', '_supabase', 'insert(', 'health_records']):
        print(f'{i}: {line.strip()[:120]}')
