import sys, json, urllib.request, time
sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, r"C:\twin\ml_prediction\src")

# Fill window with 16 calls, 0.1s apart
for i in range(16):
    req = urllib.request.Request(
        "http://localhost:5000/predict",
        data=json.dumps({"sensors":{"rpm":2320,"egt":1290,"cht":372,"oil_temp":183,"oil_press":61,"fuel_flow":8.8,"manifold":27.3}, "t":50}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
        if i >= 14:
            print("Call {}: fault={} RUL={} conf={:.2f} window_filled={}".format(i, r["fault"], r["rul_h"], r["confidence"], r["window_filled"]))
    except Exception as e:
        print("Call {} ERROR: {}".format(i, e))
    time.sleep(0.1)

# Now test high EGT
print("\n--- HIGH EGT ---")
for i in range(16):
    req = urllib.request.Request(
        "http://localhost:5000/predict",
        data=json.dumps({"sensors":{"rpm":2320,"egt":1800,"cht":372,"oil_temp":200,"oil_press":61,"fuel_flow":8.8,"manifold":27.3}, "t":60}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
        if i >= 14:
            print("Call {}: fault={} RUL={} conf={:.2f} window_filled={}".format(i, r["fault"], r["rul_h"], r["confidence"], r["window_filled"]))
    except Exception as e:
        print("Call {} ERROR: {}".format(i, e))
    time.sleep(0.1)

# Reset
req = urllib.request.Request("http://localhost:5000/reset", data=b"{}", headers={"Content-Type": "application/json"})
urllib.request.urlopen(req)
print("\nReset done")