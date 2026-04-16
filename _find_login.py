import sys
sys.stdout.reconfigure(encoding='utf-8')
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines()
for i, line in enumerate(lines, 1):
    low = line.lower()
    if any(k in low for k in ['auth', 'login', 'otp', 'send', 'secure', '_page_auth', 'def _page_login', 'sign in', 'email']):
        print(f'{i}: {line.strip()[:110]}')
