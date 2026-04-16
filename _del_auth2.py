import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').read().splitlines(keepends=True)

# Find end of old duplicate block (look for def _sidebar after the duplicate section)
start = 1282  # 0-indexed line 1283
end = start
for i in range(start, len(lines)):
    if lines[i].strip().startswith('def _sidebar('):
        end = i
        break

print(f'Deleting lines {start+1} to {end} (0-indexed {start} to {end-1})')
print(f'  First line: {lines[start][:80].rstrip()}')
print(f'  Last line:  {lines[end-1][:80].rstrip()}')
del lines[start:end]
open('app.py', 'w', encoding='utf-8').write(''.join(lines))
print(f'Done. File now has {len(lines)} lines.')
