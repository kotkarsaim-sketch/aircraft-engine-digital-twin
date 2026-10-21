import sys, json, urllib.request
sys.path.insert(0, r'C:\pylibs')
sys.path.insert(0, r'C:\twin')

# Test ML API with various conditions from the digital twin
test_cases = [
    ('HEALTHY', {'rpm':2320,'egt':1290,'cht':372,'oil_temp':183,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}, 50),
    ('HIGH_EGT', {'rpm':2320,'egt':1800,'cht':372,'oil_temp':200,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}, 60),
    ('LOW_OIL_PRESS', {'rpm':2320,'egt':1290,'cht':372,'oil_temp':200,'oil_press':25,'fuel_flow':8.8,'manifold':27.3}, 90),
    ('HIGH_CHT', {'rpm':2320,'egt':1290,'cht':520,'oil_temp':196,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}, 100),
    ('LOW_FUEL_FLOW', {'rpm':2320,'egt':1290,'cht':372,'oil_temp':183,'oil_press':61,'fuel_flow':2.5,'manifold':27.3}, 120),
    ('MULTI', {'rpm':2320,'egt':1750,'cht':372,'oil_temp':220,'oil_press':30,'fuel_flow':8.8,'manifold':27.3}, 80),
]

print('ML API Integration Test')
print('=' * 60)
for name, sensors, t in test_cases:
    # Fill window first
    for _ in range(16):
        req = urllib.request.Request('http://localhost:5000/predict',
            data=json.dumps({'sensors': sensors, 't': t}).encode(),
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req).read()
    # Get final prediction
    req = urllib.request.Request('http://localhost:5000/predict',
        data=json.dumps({'sensors': sensors, 't': t}).encode(),
        headers={'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req).read())
    rul_str = '{:.3f}h'.format(r['rul_h']) if r['rul_h'] else 'buffering...'
    print('{:<14} fault={:<9} conf={:.1%} RUL={} window={}'.format(
        name, r['fault'], r['confidence'], rul_str, r['window_filled']))
    # Reset
    urllib.request.urlopen(urllib.request.Request('http://localhost:5000/reset', data=b'{}', headers={'Content-Type':'application/json'}))

print()
print('Digital Twin Sensor Mapping:')
print('  rpm        -> rpm')
print('  egt_degf   -> egt')  
print('  cht_degf   -> cht')
print('  oil_temp_f -> oil_temp')
print('  oil_psi    -> oil_press')
print('  ff_gph     -> fuel_flow')
print('  mp_inhg    -> manifold')