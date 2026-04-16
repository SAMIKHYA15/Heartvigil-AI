import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').read().splitlines(keepends=True)
start = 2572  # 0-indexed line 2573
end = None
for i in range(start+1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('# \u2550') and len(stripped) > 20:
        end = i
        break
print(f"Deleting lines {start+1}-{end} -> {lines[start][:60].rstrip()} ... {lines[end-1][:60].rstrip()}")
del lines[start:end]
open('app.py', 'w', encoding='utf-8').write(''.join(lines))
print(f"Done. {len(lines)} lines.")
