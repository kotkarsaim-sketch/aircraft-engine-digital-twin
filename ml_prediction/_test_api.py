import json, urllib.request


def call(sensors, t, reset=False):
    if reset:
        urllib.request.urlopen(urllib.request.Request(
            "http://localhost:5000/reset", data=b"{}",
            headers={"Content-Type": "application/json"}))
    req = urllib.request.Request(
        "http://localhost:5000/predict",
        data=json.dumps({"sensors": sensors, "t": t}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


HEALTHY = {'rpm':2320,'egt':1290,'cht':372,'oil_temp':183,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}
HIGH_EGT = {'rpm':2320,'egt':1800,'cht':372,'oil_temp':200,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}
LOW_OIL = {'rpm':2320,'egt':1290,'cht':372,'oil_temp':200,'oil_press':25,'fuel_flow':8.8,'manifold':27.3}
HIGH_CHT = {'rpm':2320,'egt':1290,'cht':520,'oil_temp':196,'oil_press':61,'fuel_flow':8.8,'manifold':27.3}

# Fill window with 20 samples of one condition, then read stable RUL
for label, sensors in [("HEALTHY", HEALTHY), ("HIGH_EGT", HIGH_EGT),
                       ("LOW_OIL", LOW_OIL), ("HIGH_CHT", HIGH_CHT)]:
    call(sensors, 50, reset=True)
    for i in range(20):
        r = call(sensors, 50 + i)
    print(f"{label:<9} fault={r['fault']:<7} conf={r['confidence']:.2f} RUL={r['rul_h']:.3f}h")
