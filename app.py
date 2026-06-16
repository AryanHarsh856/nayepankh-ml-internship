import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ── Page config ──
st.set_page_config(
    page_title="NayePankh Donor Predictor",
    page_icon="🌱",
    layout="wide"
)

# ── Load model ──
@st.cache_resource
def load_model():
    with open('donor_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Header ──
st.title("🌱 NayePankh Foundation")
st.subheader("Donor Repeat Donation Predictor")
st.markdown("This ML model predicts whether a donor will donate again based on their engagement profile.")
st.divider()

# ── Sidebar inputs ──
st.sidebar.title("🧾 Enter Donor Details")

age_group = st.sidebar.selectbox(
    "Age Group",
    ['18-25', '26-35', '36-50', '51+']
)

state = st.sidebar.selectbox(
    "State",
    ['DL', 'GJ', 'HR', 'KA', 'MH', 'MP', 'RJ', 'TN', 'UP', 'WB']
)

cause_area = st.sidebar.selectbox(
    "Cause Area Interest",
    ['Education', 'Environment', 'Health', 'Other']
)

prev_donations = st.sidebar.slider(
    "Previous Donations",
    min_value=0, max_value=20, value=2
)

last_gift_amount = st.sidebar.number_input(
    "Last Gift Amount (₹)",
    min_value=0, max_value=100000, value=500, step=100
)

email_open_rate = st.sidebar.slider(
    "Email Open Rate (%)",
    min_value=0, max_value=100, value=40
)

campaigns_participated = st.sidebar.slider(
    "Campaigns Participated",
    min_value=0, max_value=10, value=2
)

volunteer_history = st.sidebar.radio(
    "Volunteer History",
    ['Yes', 'No']
)

is_social_follower = st.sidebar.radio(
    "Social Media Follower",
    ['Yes', 'No']
)

# ── Predict button ──
predict_btn = st.sidebar.button(
    "🔮 Predict Now",
    use_container_width=True
)

# ── Encode inputs ──
def encode_inputs():
    age_map = {'18-25': 0, '26-35': 1, '36-50': 2, '51+': 3}
    state_map = {'DL':0,'GJ':1,'HR':2,'KA':3,'MH':4,
                 'MP':5,'RJ':6,'TN':7,'UP':8,'WB':9}
    cause_map = {'Education':0, 'Environment':1, 'Health':2, 'Other':3}

    return pd.DataFrame([{
        'age_group': age_map[age_group],
        'state': state_map[state],
        'prev_donations': prev_donations,
        'last_gift_amount': last_gift_amount,
        'email_open_rate': email_open_rate,
        'campaigns_participated': campaigns_participated,
        'is_social_follower': 1 if is_social_follower == 'Yes' else 0,
        'volunteer_history': 1 if volunteer_history == 'Yes' else 0,
        'cause_area': cause_map[cause_area],
    }])

# ── Main content ──
if predict_btn:
    input_df = encode_inputs()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    confidence = round(probability * 100, 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction",
                  "✅ Will Donate Again" if prediction == 1
                  else "❌ Won't Donate Again")

    with col2:
        st.metric("Confidence Score", f"{confidence}%")

    with col3:
        st.metric("Risk Level",
                  "Low Risk" if confidence > 70
                  else "Medium Risk" if confidence > 50
                  else "High Risk")

    st.divider()

    # ── Confidence gauge ──
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Confidence Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Repeat Donation Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3B6D11" if prediction == 1 else "#993C1D"},
                'steps': [
                    {'range': [0, 40], 'color': '#FAECE7'},
                    {'range': [40, 70], 'color': '#FAEEDA'},
                    {'range': [70, 100], 'color': '#EAF3DE'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("Engagement Score Breakdown")
        factors = {
            'Previous Donations': min(prev_donations * 0.4, 1.0),
            'Email Open Rate': (email_open_rate / 100) * 0.25,
            'Campaigns Joined': min(campaigns_participated * 0.2, 0.8),
            'Volunteer History': 0.15 if volunteer_history == 'Yes' else 0,
            'Gift Amount': 0.1 if last_gift_amount > 500 else 0.05 if last_gift_amount > 0 else 0,
            'Social Follower': 0.06 if is_social_follower == 'Yes' else 0,
        }

        fig2 = go.Figure(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker_color='#3B6D11'
        ))
        fig2.update_layout(
            height=300,
            xaxis_title="Score",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Recommendation ──
    st.subheader("💡 Recommendation")
    if prediction == 1 and confidence > 80:
        st.success("High value donor! Send a personalised thank you message and invite them to the next campaign immediately.")
    elif prediction == 1:
        st.info("Good engagement signals. A follow-up email or campaign invite could convert them to a repeat donor.")
    elif volunteer_history == 'No':
        st.warning("Low engagement. Invite them to a free volunteering event to build a connection with the foundation first.")
    else:
        st.error("Very low engagement. Focus on improving email open rates with better subject lines and social media outreach.")

else:
    # ── Default screen ──
    st.info("👈 Fill in the donor details in the sidebar and click **Predict Now** to get started!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ML Models Trained", "4")
    col2.metric("Dataset Size", "1,200 donors")
    col3.metric("Best Accuracy", "91.7%")
    col4.metric("Best Model", "Random Forest")

    st.divider()
    st.subheader("📊 How it works")
    st.markdown("""
    1. **Enter donor details** in the sidebar
    2. **Click Predict** to run the ML model
    3. **Get instant prediction** with confidence score
    4. **Follow the recommendation** to improve donor retention
    
    ### Features used for prediction:
    - Previous donation history
    - Email open rate
    - Campaign participation
    - Volunteer history
    - Social media following
    - Last gift amount
    """)