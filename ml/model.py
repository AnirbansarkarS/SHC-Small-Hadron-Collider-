import joblib
import numpy as np
from pathlib import Path

# Load model safely so dashboard doesn't crash before it's trained
try:
    _model_path = Path(__file__).parent / "model.pkl"
    rf = joblib.load(_model_path)
except Exception:
    rf = None

def predict_stability(simulation_history):
    """
    Takes output from engine.run_simulation()
    Returns: label ('stable'/'unstable'), confidence score
    """
    if rf is None:
        return "Model Missing", 0.0

    x = np.array(simulation_history['x'])
    y = np.array(simulation_history['y'])
    r = np.sqrt(x**2 + y**2)

    # Use default 0 for features if array is empty
    if len(r) == 0:
        return "Unknown", 0.0

    # Map your simulation features to HIGGS high-level feature shape
    m_jj = np.std(r) if len(r) > 1 else 0.0
    m_jjj = np.mean(r)
    m_lv = r.max() - r.min()
    m_jlv = np.std(x) if len(x) > 1 else 0.0
    m_bb = np.std(y) if len(y) > 1 else 0.0
    m_wbb = r[-1] - r[0]
    
    if len(r) > 1:
        # np.polyfit can raise RankWarning, we just grab first coefficient
        m_wwbb = np.polyfit(range(len(r)), r, 1)[0]
    else:
        m_wwbb = 0.0

    features = np.array([[
        m_jj,         # orbit spread → m_jj proxy
        m_jjj,        # mean radius   → m_jjj proxy
        m_lv,         # drift range   → m_lv proxy
        m_jlv,        # x spread      → m_jlv proxy
        m_bb,         # y spread      → m_bb proxy
        m_wbb,        # net drift     → m_wbb proxy
        m_wwbb        # drift slope → m_wwbb proxy
    ]])

    prob = rf.predict_proba(features)[0][1]
    label = 'unstable' if prob > 0.5 else 'stable'
    return label, round(prob * 100, 1)
