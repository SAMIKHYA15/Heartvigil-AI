"""
Strip U+FE0F (variation selector) from all Python/text files.
This fixes the 'double arrow' rendering on Windows where compound emoji
show as two characters instead of one.
"""
import os

FILES = ['app.py', 'risk_agent.py', 'monitor_agent.py', 'reco_agent.py', 'data_agent.py', 'ai_helper.py']
BASE = os.path.dirname(os.path.abspath(__file__))

VARIATION_SELECTOR = '\ufe0f'  # U+FE0F - the culprit

for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        continue
    src = open(fpath, encoding='utf-8', errors='replace').read()
    if VARIATION_SELECTOR in src:
        count = src.count(VARIATION_SELECTOR)
        cleaned = src.replace(VARIATION_SELECTOR, '')
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f'Fixed {fname}: removed {count} variation selectors')
    else:
        print(f'OK    {fname}: no variation selectors found')

print()
print('Done. Restart Streamlit to see the fix.')
