import sys
sys.stdout.reconfigure(encoding='utf-8')
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines(keepends=True)

# Delete lines 1274–1459 (0-indexed: 1273–1458)
del lines[1273:1459]

open('app.py', 'w', encoding='utf-8').write(''.join(lines))
print(f'Done. File now has {len(lines)} lines.')
