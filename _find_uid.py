import sys, re
sys.stdout.reconfigure(encoding='utf-8')
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines()
for i, line in enumerate(lines, 1):
    if 'user["id"]' in line or "user['id']" in line or 'user_id' in line.lower():
        safe = line.strip()[:110]
        print(f'{i}: {safe}')
