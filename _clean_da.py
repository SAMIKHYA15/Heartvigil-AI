import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').read().splitlines(keepends=True)

# Find the dangling old function body that starts with "    def _ask_health_ai"
start = end = None
for i in range(2330, len(lines)):
    if '    def _ask_health_ai' in lines[i]:
        start = i
        break

if start:
    for i in range(start+1, len(lines)):
        s = lines[i].strip()
        if s.startswith('# \u2550') and len(s) > 20:
            end = i
            break
    print(f'Deleting lines {start+1}-{end}: {lines[start][:60].rstrip()} ... {lines[end-1][:60].rstrip()}')
    del lines[start:end]
    open('app.py', 'w', encoding='utf-8').write(''.join(lines))
    print(f'Done. {len(lines)} lines.')
else:
    print('Not found!')
