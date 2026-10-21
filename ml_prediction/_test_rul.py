import numpy as np, joblib

M = "C:/twin/ml_prediction/models/"
rul = joblib.load(M + "rul_model.joblib")
scl = joblib.load(M + "rul_scaler.joblib")

# Replicate server predict_rul exactly: rolling mean of window (1 sample here)
window = np.asarray([[2320,1290,372,183,61,8.8,27.3]], dtype=np.float32)
xr = window.mean(axis=0)
for t in [10.0, 50.0, 100.0, 200.0]:
    feat = np.hstack([xr, [t]]).reshape(1, -1)
    feat_s = scl.transform(feat)
    pred = float(rul.predict(feat_s)[0])
    print(f"t={t:>5} normalized={pred:.3f} rul_h={pred*1.0:.3f}")
