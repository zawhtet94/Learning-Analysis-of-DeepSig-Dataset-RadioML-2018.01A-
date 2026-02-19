

# =========================================================
# TELECOM FULL ANALYTICS PIPELINE — PRINT AS YOU GO VERSION
# =========================================================

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

FILE_PATH = r"D:\ZLH - Test & Learn\archive\GOLD_XYZ_OSC_POSITIVE_COMBINED.hdf5"
N = 50000
BLOCK = 100

# =========================================================
# LOAD DATA (FAST BLOCK METHOD)
# =========================================================
with h5py.File(FILE_PATH,'r') as f:
    total=f['Y'].shape[0]

    X=np.empty((N,1024,2),dtype=np.float32)
    y=np.empty(N,dtype=int)
    snr=np.empty(N)

    starts=np.linspace(0,total-BLOCK,N//BLOCK).astype(int)

    for i,s in enumerate(starts):
        sl=slice(i*BLOCK,(i+1)*BLOCK)
        X[sl]=f['X'][s:s+BLOCK]
        y[sl]=f['Y'][s:s+BLOCK]
        snr[sl]=f['Z'][s:s+BLOCK]

print("Loaded:",X.shape,y.shape,snr.shape)

# =========================================================
#  VISUAL PROOF
# =========================================================

high_snr_idx = np.where(snr >= 20)[0][0]
low_snr_idx = np.where(snr <= 0)[0][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X[high_snr_idx,:,0], X[high_snr_idx,:,1], s=2, color='green', alpha=0.4)
ax1.set_title(f"Clean (LUT-Safe) | SNR: {snr[high_snr_idx]}")
ax2.scatter(X[low_snr_idx,:,0], X[low_snr_idx,:,1], s=2, color='red', alpha=0.4)
ax2.set_title(f"Noisy (Needs Non-LUT) | SNR: {snr[low_snr_idx]}")
plt.show()


# SIGNAL METRICS
# =========================================================

I=X[:,:,0]
Q=X[:,:,1]

power=I**2+Q**2
avg_power=np.mean(power,axis=1)
peak_power=np.max(power,axis=1)
energy=np.sum(power,axis=1)
power_var=np.var(power,axis=1)

phase=np.arctan2(Q,I)
phase_var=np.var(phase,axis=1)
phase_std = np.std(phase)

# clipping=(np.abs(X)>1).mean(axis=(1,2))
clipping_mask = np.abs(X) > 1.0
clipping_rate = (np.sum(clipping_mask) / X.size) * 100



print("--- SIGNAL QUALITY REPORT ---")
print(f"Hardware Clipping Rate: {clipping_rate:.4f}%")
print(f"Phase Instability (Std Dev): {phase_std:.4f}")

print("Avg Power:",avg_power.mean())
print("Peak Power:",peak_power.mean())
print("Signal Energy:",energy.mean())
print("Clipping Rate:",clipping_rate.mean())

# =========================================================
# CHANNEL METRICS
# =========================================================
noise_floor=np.var(avg_power[snr==0]) if np.any(snr==0) else 1e-6

snr_lin=10**(np.clip(snr,-20,30)/10)
interference=np.maximum(0,power_var-snr_lin)
sinr=10*np.log10(avg_power/(noise_floor+interference+1e-9))

cqi=np.clip((snr+10)/2,1,15).astype(int)
ber=0.5*np.exp(-1.5*snr_lin)

print("Noise Floor:",noise_floor)
print("Avg SINR:",sinr.mean())
print("Avg BER:",ber.mean())
print("Avg CQI:",cqi.mean())

# =========================================================
# BUSINESS + NETWORK METRICS
# =========================================================
bit_map={0:1,1:2,2:2,3:4,4:4,5:6,6:6,7:8,8:8}
bits=np.array([bit_map[i] for i in y])

throughput=bits*1e6
pseudo_tp=throughput*(1-ber)

print("Avg Throughput:",throughput.mean()/1e6,"Mbps")
print("Pseudo Throughput:",pseudo_tp.mean()/1e6,"Mbps")

# anomaly detection
z=(avg_power-avg_power.mean())/avg_power.std()
anomaly=np.abs(z)>3

print("Anomaly Frames:",np.sum(anomaly))


# =========================================================
# NETWORK HEALTH SCORE
# =========================================================
health=(sinr/sinr.max())*0.4+(1-ber)*0.3+(1/(phase_var+1e-6)/np.max(1/(phase_var+1e-6)))*0.3
health*=100

print("Network Health Score:",health.mean())

# =========================================================
# DATAFRAME BUILD
# =========================================================

df=pd.DataFrame({
    "Mod":y,
    "SNR":snr,
    "SINR":sinr,
    "CQI":cqi,
    "BER":ber,
    "TP":throughput,
    "PseudoTP":pseudo_tp,
    "Energy":energy,
    "Peak":peak_power,
    "Interference":interference,
    "PhaseVar":phase_var,
    "Clip":clipping_rate,
    "Health":health,
    "Anomaly":anomaly
})



# =========================================================
# ROOT CAUSE ENGINE
# =========================================================
def cause(r):
    if r.Anomaly:return "Hardware Fault"
    if r.BER>0.01 and r.SINR<0:return "Extreme Noise"
    if r.Interference>1:return "Congestion"
    return "Normal"

df["Cause"]=df.apply(cause,axis=1)

print(df["Cause"].value_counts(normalize=True)*100)

# =========================================================
# NETWORK OPS METRICS
# =========================================================

grad=np.gradient(avg_power)
handover=(phase_var>1.5)&(grad<-0.05)&(clipping_rate>0.1)
congestion=np.clip((interference/(avg_power+1e-6))*100,0,100)
cell_load=100-(bits.mean()/8*100)

ces=(bits.mean()/8*40)+((1-ber.mean())*30)+((1-phase_var.mean()/np.pi)*20)+((1-clipping_rate.mean())*10)

print("Handover Risk:",handover.mean()*100,"%")
print("Congestion Prob:",congestion.mean())
print("Cell Load:",cell_load)
print("Customer Experience Score:",ces)

# =========================================================
# SNR VARIANCE PER MODULATION
# =========================================================
snr_variance=df.groupby("Mod")["SNR"].var()
print("SNR Variance: ",snr_variance)

# ========================================================
# DASHBOARD KPIs
# ========================================================

dashboard=df.groupby("Mod").agg({
"SINR":"mean",
"TP":"mean",
"BER":"mean",
"Anomaly":"sum",
"Health":"mean"
})

print(dashboard)

# =========================================================
# CHANNEL QUALITY CLASSIFIER
# =========================================================
def channel_class(v):
    if v>20:return "Excellent"
    elif v>10:return "Good"
    elif v>0:return "Fair"
    else:return "Poor"

df["ChannelClass"] = df["SINR"].apply(channel_class)

print(df["ChannelClass"].value_counts())

# =========================================================
# NOC KPIs
# =========================================================
dashboard = df.groupby("Mod").agg({
    "SINR": "mean",
    "TP": "mean",           
    "BER": "mean",
    "Anomaly": "sum",
    "Health": "mean"        
}).rename(columns={
    "Anomaly": "AnomalyCount",
    "TP": "Avg_Throughput",
    "Health": "HealthScore"
})

print("\n===== NOC KPI Dashboard =====")

# Use the new renamed column 'HealthScore' for sorting
dashboard_sorted = dashboard.sort_values("HealthScore", ascending=False).round(3)

print(dashboard_sorted)

# =========================================================
# ML MODULATION CLASSIFIER (NON-LUT ENGINE)
# =========================================================
# 1. Calculate per-frame clipping (ensure it's 50,000 rows)
# Assuming clipping_mask is (50000, 1024, 2)
clipping_rate_per_frame = clipping_mask.mean(axis=(1, 2)) 

# 2. Build the feature matrix
features = np.column_stack([
    avg_power,        # (50000,)
    peak_power,       # (50000,)
    energy,           # (50000,)
    power_var,        # (50000,)
    phase_var,        # (50000,)
    clipping_rate_per_frame  # Now (50000,) - Fixed!
])

# 3. Train and Evaluate
Xtr, Xte, ytr, yte = train_test_split(features, y, test_size=0.2, stratify=y)
model = RandomForestClassifier(n_estimators=120, max_depth=20, n_jobs=-1)
model.fit(Xtr, ytr)

model=RandomForestClassifier(n_estimators=120,max_depth=20,n_jobs=-1)
model.fit(Xtr,ytr)

pred=model.predict(Xte)

print("\nMODEL REPORT")
print(classification_report(yte,pred))
print("Confusion Matrix:\n",confusion_matrix(yte,pred))

accuracy = model.score(Xte, yte) * 100
print("\nMODEL Accuracy")
print(f"Validated Model Accuracy: {accuracy:.2f}%")

# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================

cm=confusion_matrix(yte,pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# =========================================================
# CONSTELLATION PLOTS 
# =========================================================

frame=X[100]
plt.scatter(frame[:,0],frame[:,1],s=8,alpha=.6)
plt.axhline(0);plt.axvline(0)
plt.grid(True)
plt.title("Constellation Diagram")
plt.axis("equal")
plt.show()

# Density heatmap constellation

plt.hexbin(frame[:,0],frame[:,1],gridsize=50)
plt.colorbar(label="Density")
plt.title("Constellation Density Map")
plt.axis("equal")
plt.show()

# ==========================================
# Telecom Threshold Classification
# ==========================================

def sinr_class(v):
    if v>20: return "Excellent"
    elif v>13: return "Good"
    elif v>0: return "Fair"
    else: return "Poor"

def ber_status(v):
    if v<1e-4: return "Good"
    elif v<1e-3: return "Warning"
    else: return "Critical"

def throughput_status(v, theoretical=100):
    r = v/theoretical
    if r>=0.8: return "Optimal"
    elif r>=0.5: return "Acceptable"
    else: return "Congested"

def health_severity(score):
    if score>85: return "Healthy"
    elif score>70: return "Warning"
    elif score>50: return "Degraded"
    else: return "Critical"

# Apply classifications
df["SINR_Class"] = df["SINR"].apply(sinr_class)
df["BER_Status"] = df["BER"].apply(ber_status)
df["Throughput_Status"] = df["TP"].apply(throughput_status)
df["Severity"] = df["Health"].apply(health_severity)

print("\nSignal Quality Distribution:")
print(df["SINR_Class"].value_counts())

print("\nBER Status:")
print(df["BER_Status"].value_counts())

print("\nThroughput Status:")
print(df["Throughput_Status"].value_counts())

print("\nNetwork Severity:")
print(df["Severity"].value_counts())

# ==========================================
# NOC Alert Engine
# ==========================================
alerts = df[
    (df["Severity"]=="Critical") |
    (df["BER_Status"]=="Critical") |
    (df["Throughput_Status"]=="Congested")
]

print("\n========= ACTIVE NETWORK ALERTS =========")
print(alerts[["Mod","SINR","BER","TP","Severity"]].head(20))

# =========================================================
# TELECOM AI LAYER
# =========================================================
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# Feature Set
# ==========================================
xfeatures = df[[
    "SINR",
    "TP",           
    "BER",
    "Energy",
    "Peak"          
]]

# ==========================================
# 1. CHANNEL QUALITY CLASSIFIER
# ==========================================
df["ChannelClass"] = df["SINR"].apply(
    lambda x: 3 if x>20 else 2 if x>10 else 1 if x>0 else 0
)

X_train,X_test,y_train,y_test = train_test_split(
    xfeatures,df["ChannelClass"],test_size=0.2,random_state=42
)

ch_model = RandomForestClassifier(n_estimators=120)
ch_model.fit(X_train,y_train)

pred = ch_model.predict(X_test)

print("\n===== Channel Quality Classifier =====")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# ==========================================
# 2. CONGESTION PREDICTOR
# ==========================================
df["Congestion"] = (df["TP"] < df["TP"].median()).astype(int)

X_train,X_test,y_train,y_test = train_test_split(
    xfeatures,df["Congestion"],test_size=0.2
)

cong_model = RandomForestClassifier(n_estimators=100)
cong_model.fit(X_train,y_train)

pred = cong_model.predict(X_test)

print("\n===== Congestion Predictor =====")
print(classification_report(y_test,pred))


# ==========================================
# 3. HANDOVER FAILURE PREDICTOR
# Synthetic label logic (operator-style)
# ==========================================
df["HandoverFail"] = (
    (df["SINR"]<5) &
    (df["BER"]>0.001)
).astype(int)

X_train,X_test,y_train,y_test = train_test_split(
    features,df["HandoverFail"],test_size=0.2
)

ho_model = RandomForestClassifier(n_estimators=120)
ho_model.fit(X_train,y_train)

pred = ho_model.predict(X_test)

print("\n===== Handover Failure Predictor =====")
print(classification_report(y_test,pred))


# ==========================================
# 4. ROOT CAUSE CLASSIFIER
# ==========================================
def rootcause(row):
    if row["SINR"]<5: return 0   # interference
    elif row["BER"]>0.001: return 1 # noise
    elif row["TP"]<10: return 2 # congestion
    else: return 3 # healthy

df["RootCause"] = df.apply(rootcause,axis=1)

X_train,X_test,y_train,y_test = train_test_split(
    features,df["RootCause"],test_size=0.2
)

rc_model = RandomForestClassifier(n_estimators=150)
rc_model.fit(X_train,y_train)

pred = rc_model.predict(X_test)

print("\n===== Root Cause Classifier =====")
print(classification_report(y_test,pred))


# ==========================================
# 5. SINR ESTIMATOR (Regression Model)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df[["Energy", "Peak", "PhaseVar"]], 
    df["SINR"], 
    test_size=0.2,
    random_state=42
)

sinr_model = RandomForestRegressor(n_estimators=150)
sinr_model.fit(X_train, y_train)

print("\nSINR Estimator R² Score:", sinr_model.score(X_test, y_test))


# ==========================================
# 6. ANOMALY DETECTOR
# ==========================================

iso = IsolationForest(contamination=0.03, random_state=42)
df["AI_Anomaly"] = iso.fit_predict(xfeatures)

print("\nDetected AI Anomalies:", (df["AI_Anomaly"] == -1).sum())


# ==========================================
# 7. NETWORK HEALTH SCORE (AI VERSION)
# ==========================================
df["AI_HealthScore"] = (
    df["SINR"]*0.35 +
    df["TP"]*0.25 -
    df["BER"]*200 +
    df["Energy"]*0.15
)

print("\nAvg AI Health Score:", df["AI_HealthScore"].mean())

import numpy as np
import pandas as pd

# =========================================================
# 1. SIGNAL STABILITY & INTERFERENCE
# =========================================================

# Calculating Signal Stability Index
# Higher stability indicates lower phase jitter
# Using the existing PhaseVar (phase_var) from your DataFrame
df["Stability"] = 1 / (df["PhaseVar"] + 1e-6)

# =========================================================
# 2. UPDATED NETWORK HEALTH SCORE
# =========================================================

# Normalizing metrics to create a weighted health score (0-100)
# Weighting: 40% SINR, 30% Stability, 30% Bit Error Rate (BER)
df["HealthScore"] = (
    (df["SINR"] / df["SINR"].max()) * 0.4 +
    (df["Stability"] / df["Stability"].max()) * 0.3 +
    (1 - df["BER"]) * 0.3
) * 100

# =========================================================
# 3. ADAPTIVE MODULATION ESTIMATION (SON Engine)
# =========================================================

def modulation_selector(snr_val):
    """
    Tiered selector based on Signal-to-Noise Ratio (SNR) 
    to determine optimal modulation schemes for 5G/LTE environments.
    """
    # Tier 0-2: Basic Connectivity (Robust but slow)
    if snr_val < 5:
        return "BPSK"           # 1 bit/symbol - Survival mode
    elif snr_val < 8:
        return "QPSK"           # 2 bits/symbol - Standard Control
    elif snr_val < 12:
        return "8PSK"           # 3 bits/symbol - High-speed Phase Shift
    
    # Tier 3-5: Mid-Range (Mobile Data)
    elif snr_val < 15:
        return "16QAM"          # 4 bits/symbol - Standard LTE/4G
    elif snr_val < 18:
        return "GMSK"           # Constant Envelope (IoT/2G Standard)
    elif snr_val < 22:
        return "32QAM"          # 5 bits/symbol - Specialized Tier
    
    # Tier 6-8: High Capacity (Ultra-Fast)
    elif snr_val < 25:
        return "64QAM"          # 6 bits/symbol - 5G/High-speed Wi-Fi
    elif snr_val < 30:
        return "AM-SSB/DSB"     # Analog-style specialized patterns
    else:
        return "128/256QAM"     # Peak capacity tiers

# Applying the selector to generate predicted modulation tiers
df["Predicted_Modulation"] = df["SNR"].apply(modulation_selector)

# =========================================================
# 4. FINAL TELECOM DASHBOARD SUMMARY
# =========================================================

print("\n===== Updated Network Diagnostic Summary =====")
print(df[["Mod", "SINR", "BER", "Stability", "HealthScore", "Predicted_Modulation"]].describe())

# Displaying top risk samples (Lowest Health)
print("\n===== Top 5 Critical Priority Samples =====")
print(df.sort_values("HealthScore").head(5))

# =========================================================
# SON (SELF ORGANIZING NETWORK) SYSTEM - FIXED
# =========================================================

print("\n================ SON ENGINE STARTED ================\n")

# Mapping the numeric 'Mod' to the 9 Actual Names for the Dashboard
mod_mapping = {
    0: "BPSK", 1: "QPSK", 2: "8PSK", 
    3: "16QAM", 4: "64QAM", 5: "GMSK", 
    6: "FM", 7: "AM-DSB-SC", 8: "AM-SSB-SC"
}
df["ModName"] = df["Mod"].map(mod_mapping)

# ==========================================
# 1. SELF HEALING ENGINE
# ==========================================
def self_healing(row):
    if row["SINR"] < 5:
        return "Increase Power / Adjust Tilt"
    elif row["BER"] > 0.001:
        return "Apply Noise Filtering"
    elif row["TP"] < 5:
        return "Load Balance Users"
    else:
        return "Healthy"

df["SON_Action"] = df.apply(self_healing, axis=1)

# ==========================================
# 2. SELF OPTIMIZATION ENGINE
# ==========================================
def optimize(row):
    if row["Congestion"] == 1:
        return "Shift Traffic to Neighbor Cell"
    if row["Peak"] > 1:
        return "Reduce PA Gain" # Specifically targets the clipping we found
    if row["Energy"] < 0.5:
        return "Boost Signal"
    return "Optimal"

df["Optimization_Action"] = df.apply(optimize, axis=1)

# ==========================================
# 3. SELF CONFIGURATION ENGINE
# ==========================================
def auto_config(row):
    if row["ChannelClass"] == 3:
        return "Use 256QAM"
    elif row["ChannelClass"] == 2:
        return "Use 64QAM"
    elif row["ChannelClass"] == 1:
        return "Use QPSK"
    else:
        return "Use BPSK"

df["Auto_Modulation"] = df.apply(auto_config, axis=1)

# ==========================================
# 4. NETWORK RISK SCORE
# ==========================================
# High Risk = Low SINR + High BER + Anomalies
df["RiskScore"] = (
    (5 - df["SINR"].clip(upper=5))*2 +
    df["BER"]*1000 +
    df["Congestion"]*3 +
    (df["AI_Anomaly"] == -1)*5
)

# ==========================================
# 5. SON DASHBOARD SUMMARY
# ==========================================

son_dashboard = df.groupby("ModName").agg({
    "RiskScore": "mean",
    "SINR": "mean",
    "TP": "mean",
    "BER": "mean",
    "AI_Anomaly": "count" # Using count as a proxy for Fault Frequency
}).rename(columns={"AI_Anomaly": "TotalSamples", "TP": "AvgThroughput"})

print("\n=========== SON DASHBOARD ===========")
print(son_dashboard.sort_values("RiskScore", ascending=False).round(3))

# ==========================================
# 6. TOP PROBLEM SAMPLES
# ==========================================

worst = df.sort_values("RiskScore", ascending=False).head(10)

print("\n========== TOP 10 RISK SAMPLES ==========")
print(worst[[
    "ModName", "SINR", "BER", "TP", "RiskScore",
    "SON_Action", "Optimization_Action"
]])
