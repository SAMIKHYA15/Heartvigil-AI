import sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('app.py', encoding='utf-8').read().splitlines(keepends=True)

# Find the old duplicate start (the second _page_data_agent body after the new one)
# It begins right after the new function ends with the old h2 header
start = end = None
for i in range(2580, len(lines)):
    if 'Data Agent \u2014 Assessment History' in lines[i] and 'h2' in lines[i]:
        start = i
        break

if start is None:
    print("Old code not found, checking later...")
    for i in range(2100, len(lines)):
        if 'Data Agent \u2014 Assessment History' in lines[i]:
            print(f"Found at line {i+1}: {lines[i][:80].rstrip()}")
else:
    # Find where next page function starts
    for i in range(start+1, len(lines)):
        if lines[i].strip().startswith('# \u2550') and i > start+5:
            end = i
            break
    print(f"Deleting lines {start+1} to {end} (0-indexed {start} to {end-1})")
    print(f"First: {lines[start].rstrip()[:80]}")
    print(f"Last:  {lines[end-1].rstrip()[:80]}")
    del lines[start:end]
    open('app.py', 'w', encoding='utf-8').write(''.join(lines))
    print(f"Done. File now {len(lines)} lines.")
