import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ensemble_predictor import predict_transaction

st.set_page_config(
    page_title="eCard Validator",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { 
    background-color: #EEF2F7EF; 
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #6872E4;
    padding-top: 0px;
}
            
.sub-title {
    font-size: 25px;
    font-weight: 500;
    color: #CC20AFE4;
}

.card {
    background: rgba(121, 10, 141, 0.3);
    backdrop-filter: blur(8px);
    border-radius: 5px;
    padding: 3px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

.badge-low { color: #1e8449; font-weight: 700; }
.badge-medium { color: #ca6f1e; font-weight: 700; }
.badge-high { color: #c0392b; font-weight: 700; }

[data-testid="stSidebar"] {
    background-color: #187A614C;
}

.stButton>button {
    background: #0F3479E4;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>💳 eCard_Validator</div>", unsafe_allow_html=True)
st.caption("Real-Time Credit Card Fraud Detection using Stacked Ensemble Models")

st.image("assets/img1.png", width=1000)

k1, k2, k3 = st.columns(3)
k1.metric("📊 Mode", "Real-Time")
k2.metric("🤖 Models", "3 + Meta")
k3.metric("🎯 Dataset", "ULB Credit Card")
st.divider()



st.markdown("<div class='sub-title'>💳 Why eCard_Validator???</div>", unsafe_allow_html=True)
st.caption("In today’s digital world, credit card transactions have become a common way of making payments. However, this convenience also comes with the risk of fraud. A credit card fraud detection system is essential to protect financial assets, maintain customer trust, and reduce operational losses for banks and merchants. Fraudulent transactions can result in significant financial losses if not detected in time. A reliable detection system can identify suspicious activity in real-time, preventing unauthorized transactions and reducing chargebacks. This not only saves money but also ensures that customers feel safe using their cards, strengthening their trust in financial institutions. Modern fraud detection systems use advanced technologies like machine learning and artificial intelligence to recognize unusual patterns. These systems can adapt to new fraud strategies, making them more effective than traditional rule-based methods. Moreover, they help banks comply with regulations, analyze transaction patterns, and improve overall security measures. In conclusion, a credit card fraud detection system is more than just a security tool. It is a vital part of the financial ecosystem that protects money, safeguards customer trust, and ensures the smooth operation of digital transactions. Without it, banks and customers are exposed to significant financial and reputational risks.")
st.image("assets/mo.jpg", width=1000)
st.image("assets/img2.jpg", width=1000)
st.divider()
st.markdown("<div class='sub-title'>💳 How models predict fraud?</div>", unsafe_allow_html=True)
st.caption("Credit card fraud is a major concern in the digital world, and detecting fraudulent transactions quickly is critical for banks and customers. Modern fraud detection relies on machine learning and statistical models that can analyze vast amounts of data to identify suspicious patterns. Fraud detection models work by learning from historical transaction data. Each transaction has features such as transaction amount, location, time, merchant type, and cardholder behavior. The model analyzes these features to distinguish between normal behavior and potentially fraudulent activity. For example, if a credit card is used in two different countries within a few hours, a trained model may flag it as suspicious. There are different types of models used for fraud detection. Supervised models are trained on labeled datasets where transactions are marked as fraudulent or legitimate. Algorithms such as logistic regression, decision trees, random forests, and neural networks learn patterns that separate fraud from normal transactions. Unsupervised models, on the other hand, detect anomalies without labeled data, identifying transactions that deviate significantly from typical behavior. Ensemble models, which combine multiple algorithms, are especially effective because they reduce errors and improve accuracy. Once trained, these models can predict fraud in real-time. They assign a risk score to each transaction, helping banks automatically block high-risk transactions or trigger alerts for further review. Machine learning models can also adapt over time as fraudsters change their methods, making them more resilient than static rule-based systems. In conclusion, fraud prediction models leverage data, algorithms, and continuous learning to detect suspicious activities efficiently. By analyzing transaction patterns and flagging anomalies, these models protect financial institutions and customers from losses, making them an essential tool in modern financial security.")
st.image("assets/img3.png", width=1000)

st.sidebar.header("🔐eCard_Validator")
st.sidebar.header("⚙️ System Controls")
threshold = st.sidebar.slider("Fraud Decision Threshold", 0.1, 0.9, 0.5)

if st.sidebar.button("Project files", use_container_width=True):
    st.sidebar.write("Please click here: https://github.com/mrbadbug/eCard_Validator")

st.sidebar.markdown("""
**User Manual**
- Scroll down to the Manual Transaction Prediction section to predict and validate the indivisual transactions manually.
- Scroll down to the lower end of the section Batch Transaction Analysis, upload your csv file containing the required features and get a output prediction via downloadable csv and a pychart for further analysis.
- Plays important role in validating transactions.
                    
**Developed by**
- Zakir Hussain Monir
- Daffodil International University
""")
st.divider()
st.markdown("## 🧮 Manual Transaction Prediction")
st.markdown("<div class='card'>", unsafe_allow_html=True)

features = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
manual_input = []

cols = st.columns(3)
for i, f in enumerate(features):
    col = cols[i % 3]
    if f == "Time":
        val = col.slider("Time (seconds)", 0, 502792, 0, 1)
    elif f == "Amount":
        val = col.number_input("Amount ($)", min_value=0.0, value=500.0)
    else:
        val = col.slider(f, -10.0, 10.0, 0.0, 0.01)
    manual_input.append(val)

if st.button("🔍 Predict Transaction"):
    tx = np.array(manual_input, dtype=float).reshape(1, -1)
    risk, label = predict_transaction(tx)

    st.subheader("Prediction Result")

    if risk >= threshold:
        st.error(f"🚨 FRAUD DETECTED | Risk Score: {risk:.3f}")
    elif risk >= threshold * 0.7:
        st.warning(f"⚠️ SUSPICIOUS TRANSACTION | Risk Score: {risk:.3f}")
    else:
        st.success(f"✅ SAFE TRANSACTION | Risk Score: {risk:.3f}")

    st.progress(min(int(risk * 100), 100))
    st.write(f"Risk Percentage: {risk*100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

st.markdown("## 📂 Batch Transaction Analysis")
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV (Time, V1–V28, Amount)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")

    if set(features).issubset(df.columns):
        risks = []
        predictions = []
        risk_levels = []

        for row in df[features].values:
            tx = row.reshape(1, -1)
            risk, label = predict_transaction(tx)

            if risk >= threshold:
                status = "Fraud"
                level = "High"
            elif risk >= threshold * 0.7:
                status = "Suspicious"
                level = "Suspicious"
            else:
                status = "Safe"
                level = "Low"

            risks.append(risk)
            predictions.append(status)
            risk_levels.append(level)

        df["Risk Score"] = risks
        df["Prediction"] = predictions
        df["Risk Level"] = risk_levels

        st.success("Batch prediction completed")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", len(df))
        c2.metric("Fraud Detected", (df["Prediction"] == "Fraud").sum())
        c3.metric("Avg Risk Score", round(df["Risk Score"].mean(), 3))

        st.dataframe(df.sort_values("Risk Score", ascending=False), use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "fraud_predictions.csv"
        )

        st.subheader("📊 Risk Analytics")

        fig, ax = plt.subplots()
        ax.hist(df["Risk Score"], bins=20, color="#6872E4", edgecolor="black")
        ax.set_title("Transaction Risk Distribution")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        df["Prediction"].value_counts().plot.pie(
        autopct="%1.1f%%",
        colors=["#2ecc71","#f1c40f","#e74c3c"],
        ax=ax2
        )
        ax2.set_ylabel("")
        st.pyplot(fig2)

    else:
        st.error("CSV does not contain required features")

st.markdown("</div>", unsafe_allow_html=True)
st.divider()

st.markdown("</div>", unsafe_allow_html=True)
st.divider()
st.image("assets/bg.jpg", width=1000)
st.divider()
st.caption("You can manually predict transactions or upload csv files for batch predictions, download and store the files for official uses and also see the risk analytics of the transactions using pyplot.")
