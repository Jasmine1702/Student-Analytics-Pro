
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Student Analytics Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PREMIUM RESPONSIVE THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(14,165,233,0.12), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(139,92,246,0.12), transparent 35%),
        #080F1D;
    color: #E2E8F0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101B30, #0B1425);
    border-right: 1px solid rgba(56,189,248,0.18);
}

h1, h2, h3 {
    color: #7DD3FC !important;
    font-weight: 750 !important;
}

h1 {
    background: linear-gradient(90deg, #38BDF8, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

p, label, li {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #162740, #0F1D31);
    border: 1px solid rgba(56,189,248,0.20);
    border-radius: 16px;
    padding: 18px;
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #38BDF8;
    box-shadow: 0 10px 25px rgba(14,165,233,0.18);
}

div[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #0284C7, #4F46E5);
    color: white;
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    border-color: #7DD3FC;
    box-shadow: 0 7px 20px rgba(56,189,248,0.2);
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 12px;
}

.hero-box {
    background: linear-gradient(115deg,
        rgba(14,165,233,0.17),
        rgba(79,70,229,0.17),
        rgba(15,23,42,0.8));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    padding: 28px;
    margin: 8px 0 25px 0;
    box-shadow: 0 12px 35px rgba(0,0,0,0.2);
}

.hero-title {
    font-size: 2rem;
    font-weight: 850;
    color: #E0F2FE;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #AFC4DA;
    line-height: 1.7;
}

.info-card {
    background: rgba(19,35,57,0.85);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 12px;
}

.section-label {
    color: #7DD3FC;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.footer-text {
    color: #64748B;
    font-size: 0.82rem;
    text-align: center;
    padding: 25px 0 5px 0;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(30,48,73,0.45);
    border-radius: 10px;
    padding: 9px;
    margin-bottom: 5px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(56,189,248,0.12);
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-title {
        font-size: 1.45rem;
    }

    div[data-testid="stMetric"] {
        padding: 12px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def hero(title, subtitle, eyebrow="STUDENT ANALYTICS"):
    st.markdown(f"""
    <div class="hero-box">
        <div class="section-label">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def show_table(data):
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


def style_chart(fig, ax, title=None):
    fig.patch.set_facecolor("#080F1D")
    ax.set_facecolor("#142238")
    ax.tick_params(colors="#CBD5E1")
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")

    if title:
        ax.set_title(
            title,
            color="#7DD3FC",
            fontsize=13,
            fontweight="bold",
            pad=15
        )

    for spine in ax.spines.values():
        spine.set_color("#334B68")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
        color="white"
    )

    ax.set_axisbelow(True)


# Navigation callback
def go_to_page(target):
    st.session_state["main_navigation"] = target


# ==========================================
# 4. LOAD DASHBOARD DATA
# ==========================================

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "dataset.csv")
    data = pd.read_csv(path)

    required = [
        "Student_ID", "Name", "Math",
        "Science", "English", "Attendance"
    ]

    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    data = data.drop_duplicates().copy()

    for col in ["Math", "Science", "English", "Attendance"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required).copy()

    data["Total_Marks"] = (
        data["Math"] + data["Science"] + data["English"]
    )

    data["Average_Marks"] = data["Total_Marks"] / 3

    def performance(marks):
        if marks >= 80:
            return "Excellent"
        elif marks >= 60:
            return "Good"
        elif marks >= 40:
            return "Average"
        return "Needs Improvement"

    data["Performance"] = data["Average_Marks"].apply(performance)

    data["Result"] = data.apply(
        lambda row: "Pass"
        if row["Math"] >= 40
        and row["Science"] >= 40
        and row["English"] >= 40
        and row["Attendance"] >= 75
        else "Fail",
        axis=1
    )

    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


# ==========================================
# 5. LOAD ML MODEL
# ==========================================

model_path = os.path.join(BASE_DIR, "student_model.pkl")
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.sidebar.error(f"Model loading error: {e}")


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

pages = [
    "Home",
    "Dashboard",
    "Student Search",
    "Individual Student Report",
    "Attendance Analysis",
    "Performance Analysis",
    "ML Prediction",
    "Student Risk Alerts",
    "About Project"
]

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:8px 0 18px;">
        <div style="font-size:2.5rem;">🎓</div>
        <div style="font-size:1.35rem;font-weight:850;color:#38BDF8;">
            STUDENT PRO
        </div>
        <div style="font-size:0.72rem;color:#94A3B8;">
            ACADEMIC INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        pages,
        label_visibility="visible",
        key="main_navigation",
        index=0
    )

    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <div style="color:#38BDF8;font-weight:800;">
            📊 PROJECT STATUS
        </div>
        <div style="color:#CBD5E1;font-size:0.85rem;margin-top:8px;">
            Academic analytics system
        </div>
        <div style="color:#14B8A6;font-size:0.8rem;margin-top:8px;">
            ● System available
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("🎓 Student Analytics Pro")
    st.caption("Academic Project | 2026")


# ==========================================
# 7. HOME / LANDING PAGE
# ==========================================

if page == "Home":

    hero(
        "Welcome to Student Analytics Pro",
        "An intelligent academic analytics platform designed "
        "to understand student performance, monitor attendance "
        "and support better academic decisions.",
        "YOUR ACADEMIC INTELLIGENCE PLATFORM"
    )

    st.markdown("## Explore the Platform")

    # if st.button(
    #     "🚀 Explore Dashboard",
    #     key="explore_dashboard",
    #     use_container_width=True,
    #     on_click=go_to_page,
    #     args=("Dashboard",)
    # ):
    pass

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Smart Dashboard</h3>
            <p>Explore academic statistics, charts and student records.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Dashboard →",
            key="home_dashboard",
            use_container_width=True,
            on_click=go_to_page,
            args=("Dashboard",)
        )

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 ML Prediction</h3>
            <p>Predict student outcomes using a trained Decision Tree model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore ML Prediction →",
            key="home_ml",
            use_container_width=True,
            on_click=go_to_page,
            args=("ML Prediction",)
        )

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🚨 Risk Alerts</h3>
            <p>Identify students who may need additional academic support.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Risk Alerts →",
            key="home_risk",
            use_container_width=True,
            on_click=go_to_page,
            args=("Student Risk Alerts",)
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown("""
        <div class="info-card">
            <h3>👨‍🎓 Student Reports</h3>
            <p>View individual student academic profiles.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Student Reports →",
            key="home_reports",
            use_container_width=True,
            on_click=go_to_page,
            args=("Individual Student Report",)
        )

    with c5:
        st.markdown("""
        <div class="info-card">
            <h3>📅 Attendance Analysis</h3>
            <p>Monitor attendance and identify attendance concerns.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Attendance →",
            key="home_attendance",
            use_container_width=True,
            on_click=go_to_page,
            args=("Attendance Analysis",)
        )

    with c6:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Performance Insights</h3>
            <p>Understand subject performance and student progress.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Performance →",
            key="home_performance",
            use_container_width=True,
            on_click=go_to_page,
            args=("Performance Analysis",)
        )

    st.markdown("---")

    st.markdown("## 📊 Platform Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Students", len(df))
    c2.metric(
        "Average Marks",
        f"{df['Average_Marks'].mean():.2f}"
    )
    c3.metric(
        "Average Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )

    st.info(
        "💡 You can also use the sidebar navigation "
        "to access all platform modules."
    )

    st.markdown("""
    <div class="footer-text">
        Student Analytics Pro | Turning Academic Data into Insights
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 8. SMART DASHBOARD
# ==========================================

elif page == "Dashboard":

    hero(
        "Smart Academic Dashboard",
        "Monitor student performance, academic results and attendance "
        "through an interactive dashboard.",
        "ACADEMIC OVERVIEW"
    )

    st.markdown("### 🔎 Filter Student Records")

    col1, col2 = st.columns(2)

    with col1:
        selected_performance = st.selectbox(
            "Performance Category",
            ["All"] + sorted(df["Performance"].unique().tolist())
        )

    with col2:
        selected_result = st.selectbox(
            "Student Result",
            ["All", "Pass", "Fail"]
        )

    filtered_df = df.copy()

    if selected_performance != "All":
        filtered_df = filtered_df[
            filtered_df["Performance"] == selected_performance
        ]

    if selected_result != "All":
        filtered_df = filtered_df[
            filtered_df["Result"] == selected_result
        ]

    total = len(filtered_df)
    avg = filtered_df["Average_Marks"].mean()
    attendance = filtered_df["Attendance"].mean()
    passed = int((filtered_df["Result"] == "Pass").sum())

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👨‍🎓 Total Students", total)
    c2.metric("📚 Average Marks", f"{avg:.2f}" if total else "N/A")
    c3.metric("📅 Avg. Attendance", f"{attendance:.1f}%" if total else "N/A")
    c4.metric("✅ Passed Students", passed)

    st.markdown("---")

    section_title("📋 Student Records", "Filtered academic dataset")
    show_table(filtered_df)

    if total > 0:

        col1, col2 = st.columns(2)

        with col1:
            subject_avg = filtered_df[
                ["Math", "Science", "English"]
            ].mean()

            fig, ax = plt.subplots(figsize=(7, 4))

            subject_avg.plot(
                kind="bar",
                ax=ax,
                color=["#38BDF8", "#14B8A6", "#A78BFA"]
            )

            ax.set_ylabel("Average Marks")
            ax.tick_params(axis="x", rotation=0)

            style_chart(fig, ax, "Subject-wise Average Marks")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            result_count = filtered_df["Result"].value_counts()

            fig, ax = plt.subplots(figsize=(7, 4))

            result_count.plot(
                kind="pie",
                autopct="%1.1f%%",
                colors=["#14B8A6", "#F87171"],
                startangle=90,
                textprops={"color": "white"},
                ax=ax
            )

            ax.set_ylabel("")
            fig.patch.set_facecolor("#080F1D")

            ax.set_title(
                "Pass / Fail Distribution",
                color="#7DD3FC",
                fontweight="bold"
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        top_student = filtered_df.loc[
            filtered_df["Average_Marks"].idxmax()
        ]

        st.success(
            f"🏆 Top Performer: {top_student['Name']} "
            f"with {top_student['Average_Marks']:.2f} average marks."
        )

    else:
        st.warning("No students match the selected filters.")

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        filtered_df.to_excel(
            writer,
            sheet_name="Dashboard Data",
            index=False
        )

    st.download_button(
        "📥 Download Filtered Student Data",
        data=buffer.getvalue(),
        file_name="Student_Analytics_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


# ==========================================
# 9. STUDENT SEARCH
# ==========================================

elif page == "Student Search":

    hero(
        "Student Search",
        "Find student records and explore academic details.",
        "STUDENT DIRECTORY"
    )

    student_id = st.number_input(
        "Enter Student ID",
        min_value=0,
        step=1
    )

    if st.button("🔍 Search Student", use_container_width=True):

        student = df[df["Student_ID"] == student_id]

        if not student.empty:

            record = student.iloc[0]

            st.success("Student found successfully!")

            c1, c2, c3 = st.columns(3)

            c1.metric("Average Marks", f"{record['Average_Marks']:.2f}")
            c2.metric("Attendance", f"{record['Attendance']}%")
            c3.metric("Total Marks", record["Total_Marks"])

            st.markdown(f"""
            <div class="info-card">
                <h3>{record['Name']}</h3>
                <p>Student ID: {record['Student_ID']}</p>
                <p>Performance: {record['Performance']}</p>
                <p>Final Result: {record['Result']}</p>
            </div>
            """, unsafe_allow_html=True)

            show_table(student)

        else:
            st.warning("Student ID not found.")


# ==========================================
# 10. INDIVIDUAL STUDENT REPORT
# ==========================================

elif page == "Individual Student Report":

    hero(
        "Individual Student Report",
        "View a detailed academic profile for an individual student.",
        "PERSONALIZED REPORT"
    )

    student_options = df["Student_ID"].tolist()

    selected_id = st.selectbox(
        "Select Student",
        student_options,
        format_func=lambda x: (
            f"{df.loc[df['Student_ID'] == x, 'Name'].iloc[0]} "
            f"(ID: {x})"
        )
    )

    record = df[df["Student_ID"] == selected_id].iloc[0]

    st.markdown("---")
    st.markdown(f"## 👨‍🎓 {record['Name']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Student ID", record["Student_ID"])
    c2.metric("Average Marks", f"{record['Average_Marks']:.2f}")
    c3.metric("Attendance", f"{record['Attendance']}%")
    c4.metric("Result", record["Result"])

    st.markdown("---")
    section_title("📚 Subject-wise Performance")

    subject_data = pd.DataFrame({
        "Subject": ["Math", "Science", "English"],
        "Marks": [
            record["Math"],
            record["Science"],
            record["English"]
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            subject_data["Subject"],
            subject_data["Marks"],
            color=["#38BDF8", "#14B8A6", "#A78BFA"]
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Marks")

        style_chart(fig, ax, "Subject Performance")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("### Academic Summary")

        st.markdown(f"""
        <div class="info-card">
            <p><b>Total Marks:</b> {record['Total_Marks']}</p>
            <p><b>Average Marks:</b> {record['Average_Marks']:.2f}</p>
            <p><b>Performance:</b> {record['Performance']}</p>
            <p><b>Attendance:</b> {record['Attendance']}%</p>
            <p><b>Final Result:</b> {record['Result']}</p>
        </div>
        """, unsafe_allow_html=True)

    report = pd.DataFrame([record])

    st.download_button(
        "📥 Download Individual Student Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"Student_Report_{selected_id}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# 11. ATTENDANCE ANALYSIS
# ==========================================

elif page == "Attendance Analysis":

    hero(
        "Attendance Analysis",
        "Monitor attendance and identify students who may need support.",
        "ATTENDANCE INTELLIGENCE"
    )

    avg_attendance = df["Attendance"].mean()
    low_count = int((df["Attendance"] < 75).sum())

    c1, c2 = st.columns(2)

    c1.metric("📅 Average Attendance", f"{avg_attendance:.2f}%")
    c2.metric("⚠️ Low Attendance Students", low_count)

    fig, ax = plt.subplots(figsize=(11, 5))

    bar_colors = [
        "#F87171" if value < 75 else "#38BDF8"
        for value in df["Attendance"]
    ]

    ax.bar(
        df["Name"],
        df["Attendance"],
        color=bar_colors
    )

    ax.axhline(
        75,
        linestyle="--",
        color="#FBBF24",
        linewidth=2,
        label="75% Attendance Requirement"
    )

    ax.set_ylabel("Attendance (%)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=40)
    ax.legend()

    style_chart(fig, ax, "Student-wise Attendance")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    low = df[df["Attendance"] < 75]

    section_title("⚠️ Students with Low Attendance")

    if not low.empty:
        show_table(low[["Student_ID", "Name", "Attendance"]])
    else:
        st.success("No students with low attendance.")


# ==========================================
# 12. PERFORMANCE ANALYSIS
# ==========================================

elif page == "Performance Analysis":

    hero(
        "Performance Analysis",
        "Understand performance categories and recognize top performers.",
        "ACADEMIC INSIGHTS"
    )

    counts = df["Performance"].value_counts()

    colors = {
        "Excellent": "#14B8A6",
        "Good": "#38BDF8",
        "Average": "#A78BFA",
        "Needs Improvement": "#F87171"
    }

    fig, ax = plt.subplots(figsize=(9, 4))

    counts.plot(
        kind="bar",
        ax=ax,
        color=[colors.get(x, "#38BDF8") for x in counts.index]
    )

    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=20)

    style_chart(fig, ax, "Student Performance Distribution")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    section_title("🏆 Top Performing Students")

    top_students = df.sort_values(
        by="Average_Marks",
        ascending=False
    ).head(5)

    show_table(
        top_students[
            ["Student_ID", "Name", "Average_Marks", "Performance"]
        ]
    )

    section_title("📌 Students Needing Improvement")

    weak_students = df[
        df["Performance"] == "Needs Improvement"
    ]

    if not weak_students.empty:
        show_table(weak_students)
    else:
        st.success("No students in this category.")


# ==========================================
# 13. BETTER ML PREDICTION
# ==========================================

elif page == "ML Prediction":

    hero(
        "Student Result Prediction",
        "Predict academic outcomes using a trained Decision Tree "
        "Machine Learning model.",
        "MACHINE LEARNING LAB"
    )

    if model is None:
        st.error("Model file student_model.pkl could not be loaded.")
        st.stop()

    feature_columns = [
        "studytime", "failures", "absences",
        "G1", "G2", "age", "health", "freetime"
    ]

    feature_names = [
        "Study Time", "Past Failures", "Absences",
        "First Period Grade", "Second Period Grade",
        "Age", "Health", "Free Time"
    ]

    st.markdown("### 📊 Model Information")

    c1, c2 = st.columns(2)

    c1.metric("ML Algorithm", "Decision Tree")
    c2.metric("Input Features", len(feature_columns))

    st.markdown("---")
    section_title("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        if len(importance) == len(feature_names):

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(
                by="Importance",
                ascending=True
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"],
                color="#38BDF8"
            )

            ax.set_xlabel("Feature Importance")
            style_chart(fig, ax, "Factors Affecting Prediction")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    section_title("📈 Model Evaluation")

    try:
        ml_path = os.path.join(
            BASE_DIR, "student_data", "student-mat.txt"
        )

        ml_data = pd.read_csv(ml_path, sep=";")

        ml_data["Result"] = ml_data["G3"].apply(
            lambda marks: "Pass" if marks >= 10 else "Fail"
        )

        X = ml_data[feature_columns]
        y = ml_data["Result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["Fail", "Pass"]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"],
            ax=ax,
            linewidths=1,
            linecolor="#334B68"
        )

        ax.set_xlabel("Predicted Result", color="white")
        ax.set_ylabel("Actual Result", color="white")
        ax.set_title("Confusion Matrix", color="#7DD3FC")
        ax.tick_params(colors="white")

        fig.patch.set_facecolor("#080F1D")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Evaluation uses the project's 80/20 stratified test split. "
            "It assumes the saved model was trained on the matching split."
        )

    except Exception as e:
        st.warning(f"Model evaluation unavailable: {e}")

    st.markdown("---")
    section_title("📝 Enter Student Information")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Less than 2 hours",
                    2: "2 to 5 hours",
                    3: "5 to 10 hours",
                    4: "More than 10 hours"
                }[x]
            )

            failures = st.number_input(
                "Past Class Failures",
                min_value=0,
                max_value=4,
                value=0
            )

            absences = st.number_input(
                "Number of Absences",
                min_value=0,
                max_value=100,
                value=0
            )

            age = st.number_input(
                "Student Age",
                min_value=15,
                max_value=25,
                value=18
            )

        with col2:

            G1 = st.slider("First Period Grade", 0, 20, 10)
            G2 = st.slider("Second Period Grade", 0, 20, 10)
            health = st.slider("Health (1-5)", 1, 5, 3)
            freetime = st.slider("Free Time (1-5)", 1, 5, 3)

        predict_button = st.form_submit_button(
            "🚀 Predict Student Result",
            use_container_width=True
        )

    if predict_button:

        input_data = pd.DataFrame(
            [[
                studytime, failures, absences,
                G1, G2, age, health, freetime
            ]],
            columns=feature_columns
        )

        try:

            prediction = model.predict(input_data)[0]

            st.markdown("---")
            section_title("🎯 Prediction Result")

            if prediction == "Pass":
                st.success("🎉 Predicted Result: PASS")
            else:
                st.error("⚠️ Predicted Result: FAIL")

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                probability_df = pd.DataFrame({
                    "Result": model.classes_,
                    "Probability": probabilities
                })

                st.markdown("### Prediction Probability")

                for _, row in probability_df.iterrows():

                    st.write(
                        f"{row['Result']}: {row['Probability'] * 100:.2f}%"
                    )

                    st.progress(float(row["Probability"]))

            st.info(
                "This prediction is an estimate from the trained model, "
                "not a guaranteed academic result."
            )

            st.caption(
                "The model uses earlier grades (G1 and G2), "
                "study time, absences and other student attributes."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==========================================
# 14. STUDENT RISK ALERTS
# ==========================================

elif page == "Student Risk Alerts":

    hero(
        "Student Risk Alerts",
        "Identify students who may need academic attention "
        "using attendance and marks.",
        "EARLY WARNING SYSTEM"
    )

    def detect_risk(row):

        if (
            row["Average_Marks"] < 40
            or row["Attendance"] < 60
        ):
            return "High Risk"

        elif (
            row["Average_Marks"] < 60
            or row["Attendance"] < 75
        ):
            return "Medium Risk"

        return "Low Risk"

    def suggestion(row):

        tips = []

        if row["Attendance"] < 75:
            tips.append("Improve attendance")

        if row["Average_Marks"] < 60:
            tips.append("Provide additional academic support")

        if row["Math"] < 40:
            tips.append("Focus on Mathematics")

        if row["Science"] < 40:
            tips.append("Focus on Science")

        if row["English"] < 40:
            tips.append("Focus on English")

        if not tips:
            return "Maintain current performance"

        return " • ".join(tips)

    risk_df = df.copy()

    risk_df["Risk_Level"] = risk_df.apply(
        detect_risk,
        axis=1
    )

    risk_df["Suggested_Action"] = risk_df.apply(
        suggestion,
        axis=1
    )

    high = int((risk_df["Risk_Level"] == "High Risk").sum())
    medium = int((risk_df["Risk_Level"] == "Medium Risk").sum())
    low = int((risk_df["Risk_Level"] == "Low Risk").sum())

    c1, c2, c3 = st.columns(3)

    c1.metric("🔴 High Risk", high)
    c2.metric("🟡 Medium Risk", medium)
    c3.metric("🟢 Low Risk", low)

    st.markdown("---")
    section_title("🚨 Priority Alerts")

    high_risk = risk_df[
        risk_df["Risk_Level"] == "High Risk"
    ]

    if not high_risk.empty:

        for _, row in high_risk.iterrows():

            st.error(
                f"🔴 {row['Name']} | "
                f"Attendance: {row['Attendance']}% | "
                f"Average: {row['Average_Marks']:.2f}"
            )

            st.caption(
                f"Suggested action: {row['Suggested_Action']}"
            )

    else:
        st.success("No high-risk students detected.")

    st.markdown("---")
    section_title("📊 Risk Distribution")

    risk_counts = risk_df["Risk_Level"].value_counts()

    risk_order = ["High Risk", "Medium Risk", "Low Risk"]

    risk_counts = risk_counts.reindex(
        risk_order,
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    risk_counts.plot(
        kind="bar",
        ax=ax,
        color=["#F87171", "#FBBF24", "#14B8A6"]
    )

    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=0)

    style_chart(fig, ax, "Student Risk Distribution")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    section_title("📋 Detailed Risk Report")

    show_table(
        risk_df[
            [
                "Student_ID",
                "Name",
                "Average_Marks",
                "Attendance",
                "Risk_Level",
                "Suggested_Action"
            ]
        ]
    )

    st.download_button(
        "📥 Download Risk Report",
        data=risk_df.to_csv(index=False).encode("utf-8"),
        file_name="Student_Risk_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Risk categories are based on project-defined marks "
        "and attendance rules. They are not ML predictions."
    )


# ==========================================
# 15. ABOUT PROJECT & ML DETAILS
# ==========================================

elif page == "About Project":

    hero(
        "About Student Analytics Pro",
        "A student academic performance analysis and prediction system "
        "built using Python, data analysis and machine learning.",
        "PROJECT INFORMATION"
    )

    section_title("🎯 Project Objective")

    st.markdown("""
    <div class="info-card">
        Student Analytics Pro helps explore academic records, analyze
        student performance, monitor attendance and estimate student
        results using a Machine Learning model.
    </div>
    """, unsafe_allow_html=True)

    section_title("⚙️ Technology Stack")

    technologies = [
        ("🐍 Python", "Core programming language"),
        ("📊 Pandas", "Data handling and analysis"),
        ("📈 Matplotlib", "Data visualization"),
        ("🎨 Seaborn", "Confusion matrix visualization"),
        ("🤖 Scikit-learn", "Machine Learning model"),
        ("⚡ Streamlit", "Interactive web application"),
        ("📦 Joblib", "Saved ML model loading")
    ]

    for name, description in technologies:

        st.markdown(f"""
        <div class="info-card">
            <b style="color:#38BDF8;">{name}</b>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    section_title("🤖 Machine Learning Details")

    ml_info = [
        ("Algorithm", "Decision Tree Classifier"),
        ("Dataset", "UCI Student Performance - Mathematics"),
        ("Target", "Pass / Fail"),
        ("Training Split", "80% Training / 20% Testing"),
        ("Random State", "42"),
        ("Features", "Study Time, Failures, Absences, G1, G2, Age, Health, Free Time")
    ]

    for label, value in ml_info:
        st.markdown(f"**{label}:** {value}")

    if model is not None:

        st.success("Trained ML model loaded successfully.")

        if hasattr(model, "max_depth"):
            st.write(f"Decision Tree Maximum Depth: {model.max_depth}")

    else:
        st.warning("ML model file is not currently available.")

    st.markdown("---")
    section_title("📂 Project Modules")

    modules = [
        "Home and Landing Page",
        "Smart Academic Dashboard",
        "Student Search",
        "Individual Student Report",
        "Attendance Analysis",
        "Performance Analysis",
        "Machine Learning Prediction",
        "Student Risk Alerts"
    ]

    for module in modules:
        st.markdown(f"✅ {module}")

    st.markdown("---")
    section_title("📌 Important Project Notes")

    st.info(
        "The ML model uses the UCI Mathematics dataset. "
        "The dashboard uses a separate student records CSV file."
    )

    st.warning(
        "ML predictions are estimates, not guarantees. "
        "Risk alerts are generated using predefined academic rules."
    )

    st.markdown("""
    <div class="info-card">
        <h3>🎓 Student Analytics Pro</h3>
        <p>Academic Performance • Machine Learning • Data Insights</p>
        <p>Built with Python and Streamlit | 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 16. GLOBAL FOOTER
# ==========================================

st.markdown("""
<div class="footer-text">
    🎓 STUDENT ANALYTICS PRO
    <br>
    Academic Performance • Machine Learning • Data Insights
    <br>
    Built with Python and Streamlit | 2026
</div>
""", unsafe_allow_html=True)
