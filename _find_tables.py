import sys
sys.stdout.reconfigure(encoding='utf-8')
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines()
for i, line in enumerate(lines, 1):
    low = line.lower()
    if any(k in low for k in ['dataframe', 'st.table', 'disp_cols', 'raw', 'history', 'df_disp', 'assessment record']):
        print(f'{i}: {line.strip()[:120]}')
