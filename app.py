import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Optional interactive plotting with graceful fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt
    import seaborn as sns

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (MODERN, CLEAN, RESPONSIVE)
# ==========================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    @media (prefers-color-scheme: dark) {
        .card {
            background: #1E293B;
            border-color: #334155;
        }
    }

    /* Result Card - Placed */
    .result-placed {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 18px;
        padding: 1.8rem;
        color: #065F46;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(16, 185, 129, 0.2);
    }
    /* Result Card - Not Placed */
    .result-unplaced {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 2px solid #F43F5E;
        border-radius: 18px;
        padding: 1.8rem;
        color: #9F1239;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(244, 63, 94, 0.2);
    }

    .badge-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .badge-success {
        background-color: #10B981;
        color: white;
    }
    .badge-danger {
        background-color: #F43F5E;
        color: white;
    }

    /* Metric Badges */
    .stat-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #312E81;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
    }

    /* St tabs header */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL TRAINING & DATA LOADING
# ==========================================
@st.cache_resource
def load_and_train_model():
    dataset_path = "placement-dataset.csv"
    
    if not os.path.exists(dataset_path):
        # Fallback dataset if file isn't present
        url = "https://raw.githubusercontent.com/campusx-official/placement-project-logistic-regression/main/placement.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(dataset_path)
    
    # Clean up column names
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col or col == ""], errors="ignore")
    
    X = df[['cgpa', 'iq']]
    y = df['placement']
    
    # Exact split from the model notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "model": model,
        "data": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "cm": cm,
        "report": report
    }

artifacts = load_and_train_model()
model = artifacts["model"]
df = artifacts["data"]
accuracy = artifacts["accuracy"]
cm = artifacts["cm"]


# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎓 Student Placement Predictor</div>
    <p class="hero-subtitle">
        An AI-powered decision support tool predicting campus placement probability based on student academic performance (CGPA) and cognitive ability (IQ).
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR - INPUT CONTROLS & PRESETS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Student Profile Input")
    st.caption("Adjust the student metrics below or select a quick preset scenario:")
    
    # Preset Profiles
    st.markdown("**Quick Preset Profiles:**")
    col_p1, col_p2 = st.columns(2)
    
    # Initialize session state for inputs if not present
    if "cgpa_input" not in st.session_state:
        st.session_state.cgpa_input = 7.2
    if "iq_input" not in st.session_state:
        st.session_state.iq_input = 120.0
        
    with col_p1:
        if st.button("🌟 Star Performer", use_container_width=True):
            st.session_state.cgpa_input = 8.2
            st.session_state.iq_input = 145.0
        if st.button("📚 Balanced", use_container_width=True):
            st.session_state.cgpa_input = 6.8
            st.session_state.iq_input = 115.0
            
    with col_p2:
        if st.button("💡 High IQ / Low CGPA", use_container_width=True):
            st.session_state.cgpa_input = 5.2
            st.session_state.iq_input = 175.0
        if st.button("⚠️ Needs Focus", use_container_width=True):
            st.session_state.cgpa_input = 4.8
            st.session_state.iq_input = 95.0

    st.markdown("---")
    
    # CGPA Slider & Number Input
    cgpa = st.slider(
        "Cumulative Grade Point Average (CGPA)",
        min_value=3.0,
        max_value=10.0,
        value=float(st.session_state.cgpa_input),
        step=0.1,
        help="Academic CGPA on a scale of 0 to 10"
    )
    st.session_state.cgpa_input = cgpa
    
    # IQ Slider & Number Input
    iq = st.slider(
        "Intelligence Quotient (IQ Score)",
        min_value=40.0,
        max_value=240.0,
        value=float(st.session_state.iq_input),
        step=1.0,
        help="Cognitive IQ evaluation score"
    )
    st.session_state.iq_input = iq

    st.markdown("---")
    st.markdown("### 📌 Model Information")
    st.markdown(f"""
    - **Algorithm:** Logistic Regression
    - **Test Accuracy:** `{accuracy * 100:.1f}%`
    - **Dataset Size:** `100 records`
    - **Features:** `CGPA`, `IQ`
    """)
    st.info("💡 **Insight:** In this dataset, CGPA carries the highest weight towards determining placement success.")


# ==========================================
# MAIN CONTENT TABS
# ==========================================
tab_predict, tab_boundary, tab_metrics, tab_data = st.tabs([
    "🎯 Prediction & Insights", 
    "🗺️ Visual Decision Boundary", 
    "📈 Model Evaluation", 
    "📂 Dataset Explorer"
])

# ------------------------------------------
# TAB 1: PREDICTION & INSIGHTS
# ------------------------------------------
with tab_predict:
    col_input_sum, col_pred_card = st.columns([1, 1.3], gap="large")
    
    # Compute Model Prediction
    input_data = pd.DataFrame({"cgpa": [cgpa], "iq": [iq]})
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    prob_unplaced = probabilities[0] * 100
    prob_placed = probabilities[1] * 100
    
    with col_input_sum:
        st.markdown("### 📋 Student Evaluation Profile")
        st.write("Current profile parameters being evaluated by the ML model:")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{cgpa:.1f}</div>
                <div class="stat-label">CGPA (Score)</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{int(iq)}</div>
                <div class="stat-label">IQ Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Decision Boundary Distance & Threshold calculation
        # logit = w_cgpa * cgpa + w_iq * iq + intercept
        w_cgpa = model.coef_[0][0]
        w_iq = model.coef_[0][1]
        bias = model.intercept_[0]
        
        # Min CGPA needed at this IQ to reach 50% probability (where logit >= 0):
        # w_cgpa * cgpa_target + w_iq * iq + bias = 0
        min_cgpa_needed = -(w_iq * iq + bias) / w_cgpa
        
        st.markdown("#### 🎯 Placement Threshold Analysis")
        if prediction == 1:
            st.success(f"✅ At an IQ of **{int(iq)}**, a student typically needs a minimum CGPA of **{max(0.0, min_cgpa_needed):.2f}** to qualify for placement. This profile comfortably clears the threshold.")
        else:
            cgpa_diff = min_cgpa_needed - cgpa
            st.warning(f"📈 To achieve a placement probability $\ge 50\%$ at IQ **{int(iq)}**, this student needs to raise their CGPA by approximately **+{cgpa_diff:.2f}** points (Target: **{min_cgpa_needed:.2f}**).")

    with col_pred_card:
        st.markdown("### 🔮 Predicted Outcome")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-placed">
                <div class="badge-pill badge-success">High Probability</div>
                <h2 style="margin: 0.5rem 0; font-weight: 800; font-size: 2rem; color: #065F46;">🎉 LIKELY TO BE PLACED</h2>
                <p style="font-size: 1.05rem; margin-bottom: 1.2rem; color: #047857;">
                    The student demonstrates strong academic and cognitive attributes consistent with successfully placed candidates.
                </p>
                <div style="background: white; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 6px; color: #065F46;">
                        <span>Placement Confidence</span>
                        <span>{prob_placed:.2f}%</span>
                    </div>
                    <div style="background: #E5E7EB; border-radius: 9999px; height: 14px; overflow: hidden;">
                        <div style="background: #10B981; width: {prob_placed}%; height: 100%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unplaced">
                <div class="badge-pill badge-danger">Attention Required</div>
                <h2 style="margin: 0.5rem 0; font-weight: 800; font-size: 2rem; color: #9F1239;">⚠️ UNLIKELY TO BE PLACED</h2>
                <p style="font-size: 1.05rem; margin-bottom: 1.2rem; color: #BE123C;">
                    Current metrics fall below the historical campus placement decision boundary. Focus on core academic grades is recommended.
                </p>
                <div style="background: white; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 6px; color: #9F1239;">
                        <span>Risk Factor (Not Placed)</span>
                        <span>{prob_unplaced:.2f}%</span>
                    </div>
                    <div style="background: #E5E7EB; border-radius: 9999px; height: 14px; overflow: hidden;">
                        <div style="background: #F43F5E; width: {prob_unplaced}%; height: 100%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Probability Breakdown gauge or bar
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Placed Probability", value=f"{prob_placed:.1f}%", delta=f"{prob_placed - 50:.1f}% vs baseline")
        with m2:
            st.metric(label="Not Placed Probability", value=f"{prob_unplaced:.1f}%", delta=f"{prob_unplaced - 50:.1f}% vs baseline", delta_color="inverse")


# ------------------------------------------
# TAB 2: DECISION BOUNDARY & DATA VISUALIZATION
# ------------------------------------------
with tab_boundary:
    st.markdown("### 🗺️ Logistic Regression Decision Boundary")
    st.caption("Explore how the trained model separates placed vs non-placed students, and see where the current student stands in relation to the threshold line.")
    
    if HAS_PLOTLY:
        # Generate grid for decision boundary line
        cgpa_range = np.linspace(df['cgpa'].min() - 0.5, df['cgpa'].max() + 0.5, 200)
        # Decision boundary: w1 * cgpa + w2 * iq + b = 0 => iq = -(w1 * cgpa + b) / w2
        iq_boundary = -(w_cgpa * cgpa_range + bias) / w_iq
        
        # Valid boundary points within realistic range
        valid_idx = (iq_boundary >= 20) & (iq_boundary <= 250)
        
        fig = go.Figure()

        # Placed points
        df_placed = df[df['placement'] == 1]
        fig.add_trace(go.Scatter(
            x=df_placed['cgpa'],
            y=df_placed['iq'],
            mode='markers',
            name='Placed (Data)',
            marker=dict(size=9, color='#10B981', symbol='circle', opacity=0.85, line=dict(width=1, color='#065F46')),
            hovertemplate="<b>Placed Student</b><br>CGPA: %{x:.2f}<br>IQ: %{y:.1f}<extra></extra>"
        ))

        # Not Placed points
        df_unplaced = df[df['placement'] == 0]
        fig.add_trace(go.Scatter(
            x=df_unplaced['cgpa'],
            y=df_unplaced['iq'],
            mode='markers',
            name='Not Placed (Data)',
            marker=dict(size=9, color='#F43F5E', symbol='x', opacity=0.85, line=dict(width=1, color='#881337')),
            hovertemplate="<b>Not Placed Student</b><br>CGPA: %{x:.2f}<br>IQ: %{y:.1f}<extra></extra>"
        ))

        # Boundary Line
        fig.add_trace(go.Scatter(
            x=cgpa_range[valid_idx],
            y=iq_boundary[valid_idx],
            mode='lines',
            name='Decision Boundary (P=0.5)',
            line=dict(color='#4F46E5', width=3, dash='dash'),
            hoverinfo='skip'
        ))

        # Current Student marker
        fig.add_trace(go.Scatter(
            x=[cgpa],
            y=[iq],
            mode='markers+text',
            name='Current Student',
            text=["📍 Candidate"],
            textposition="top center",
            marker=dict(
                size=18, 
                color='#F59E0B' if prediction == 1 else '#9333EA', 
                symbol='star',
                line=dict(width=2, color='#FFFFFF')
            ),
            hovertemplate=f"<b>Current Student</b><br>CGPA: {cgpa:.2f}<br>IQ: {iq:.1f}<br>Predicted: {'Placed 🎉' if prediction == 1 else 'Not Placed ⚠️'}<extra></extra>"
        ))

        fig.update_layout(
            title="Student Distribution & Classification Boundary",
            xaxis_title="CGPA (Cumulative Grade Point Average)",
            yaxis_title="IQ (Intelligence Quotient)",
            template="plotly_white",
            height=550,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='cgpa', y='iq', hue='placement', palette={0: '#F43F5E', 1: '#10B981'}, s=80, ax=ax)
        
        # Decision boundary
        cgpa_range = np.linspace(df['cgpa'].min() - 0.5, df['cgpa'].max() + 0.5, 200)
        iq_boundary = -(w_cgpa * cgpa_range + bias) / w_iq
        ax.plot(cgpa_range, iq_boundary, color='#4F46E5', linestyle='--', linewidth=2, label='Decision Boundary')
        
        # Current candidate
        ax.scatter([cgpa], [iq], color='gold', s=250, edgecolor='black', marker='*', label='Current Candidate')
        
        ax.set_ylim(20, 250)
        ax.set_title("CGPA vs IQ Placement Boundary", fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)


# ------------------------------------------
# TAB 3: MODEL EVALUATION & METRICS
# ------------------------------------------
with tab_metrics:
    st.markdown("### 📈 Model Evaluation & Performance")
    st.caption("Performance metrics on the stratified test split (20% of data, 20 students):")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")
    with col_m2:
        st.metric("Placed Precision", f"{artifacts['report']['1']['precision'] * 100:.1f}%")
    with col_m3:
        st.metric("Placed Recall", f"{artifacts['report']['1']['recall'] * 100:.1f}%")
    with col_m4:
        st.metric("F1-Score", f"{artifacts['report']['1']['f1-score'] * 100:.1f}%")
        
    st.markdown("---")
    
    col_cm, col_formula = st.columns([1.1, 1], gap="large")
    
    with col_cm:
        st.markdown("#### 🧩 Confusion Matrix")
        if HAS_PLOTLY:
            cm_z = cm
            cm_x = ['Predicted: Not Placed (0)', 'Predicted: Placed (1)']
            cm_y = ['Actual: Not Placed (0)', 'Actual: Placed (1)']
            
            fig_cm = px.imshow(
                cm_z,
                text_auto=True,
                x=cm_x,
                y=cm_y,
                color_continuous_scale="Blues",
                aspect="auto"
            )
            fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                        xticklabels=['Not Placed', 'Placed'],
                        yticklabels=['Not Placed', 'Placed'])
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            
    with col_formula:
        st.markdown("#### 🧮 Model Formulation")
        st.markdown("The model computes probability using the **Sigmoid (Logistic) Function**:")
        st.latex(r"P(\text{Placement} = 1) = \frac{1}{1 + e^{-z}}")
        st.markdown("Where the linear combination $z$ is parameterized as:")
        st.latex(r"z = w_1 \cdot \text{CGPA} + w_2 \cdot \text{IQ} + b")
        st.markdown("**Fitted Coefficients:**")
        st.markdown(f"""
        - $w_1$ (CGPA Weight): `{w_cgpa:.4f}`
        - $w_2$ (IQ Weight): `{w_iq:.4f}`
        - $b$ (Intercept / Bias): `{bias:.4f}`
        """)
        st.info("The positive coefficient on CGPA indicates that higher CGPA substantially boosts placement odds.")


# ------------------------------------------
# TAB 4: DATASET EXPLORER
# ------------------------------------------
with tab_data:
    st.markdown("### 📂 Placement Dataset Overview")
    st.caption("Historical campus recruitment dataset used for training the model:")
    
    col_d1, col_d2 = st.columns([1.5, 1])
    with col_d1:
        st.dataframe(df, use_container_width=True, height=360)
    with col_d2:
        st.markdown("#### 📊 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
    st.markdown("#### 🔍 Filter Dataset")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        placement_filter = st.multiselect("Filter by Placement Status", options=[0, 1], default=[0, 1], format_func=lambda x: "Placed (1)" if x == 1 else "Not Placed (0)")
    with f_col2:
        min_cgpa_filter = st.slider("Filter by Minimum CGPA", min_value=float(df['cgpa'].min()), max_value=float(df['cgpa'].max()), value=float(df['cgpa'].min()))
        
    filtered_df = df[(df['placement'].isin(placement_filter)) & (df['cgpa'] >= min_cgpa_filter)]
    st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** records:")
    st.dataframe(filtered_df, use_container_width=True, height=220)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.9rem; padding: 10px 0;">
    🎓 Student Placement Prediction App • Built with Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
