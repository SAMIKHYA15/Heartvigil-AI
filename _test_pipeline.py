import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, '.')

errors = []

# Test 1: All imports
try:
    from data_agent import run_data_agent, validate_fields, SAFE_RANGES
    from risk_agent import run_risk_agent
    from monitor_agent import run_monitor_agent
    from reco_agent import run_reco_agent
    from supabase_client import get_supabase, get_admin_supabase
    print('OK: All imports')
except Exception as e:
    errors.append(f'IMPORT: {e}')

# Test 2: Supabase admin client
uid = None
try:
    sb = get_admin_supabase()
    r  = sb.table('users').select('id,email').limit(1).execute()
    uid = r.data[0]['id'] if r.data else None
    print(f'OK: Admin client, uid={uid}')
except Exception as e:
    errors.append(f'SUPABASE: {e}')

# Test 3: Risk agent
hd = {'age':55,'sex':1,'cp':3,'trestbps':130,'chol':250,'fbs':0,
      'restecg':0,'thalach':140,'exang':1,'oldpeak':2.0,'slope':1,'ca':1,'thal':2}
try:
    ro = run_risk_agent(hd, use_groq=False)
    print(f"OK: Risk agent => {ro['risk_label']} {ro['risk_pct']:.1f}%")
except Exception as e:
    errors.append(f'RISK_AGENT: {e}')

# Test 4: Data agent save (then delete)
if uid:
    try:
        result = run_data_agent(get_admin_supabase(), uid, hd,
                                risk_score=0.45, risk_label='MEDIUM', source='test')
        if result['success']:
            # clean up test record
            rec = get_admin_supabase().table('health_records').select('id').eq('user_id', uid).order('created_at', desc=True).limit(1).execute()
            if rec.data:
                get_admin_supabase().table('health_records').delete().eq('id', rec.data[0]['id']).execute()
            print('OK: Data agent save + cleanup')
        else:
            errors.append(f"DATA_AGENT: {result['message']} errors={result.get('errors')}")
    except Exception as e:
        import traceback
        errors.append(f'DATA_AGENT_EXCEPTION: {e}')
        traceback.print_exc()

print()
if errors:
    print('ERRORS FOUND:')
    for e in errors:
        print(' -', e)
else:
    print('All tests passed! Database and agents are working.')
