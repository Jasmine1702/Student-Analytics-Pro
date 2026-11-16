import os
import shutil
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
BACKUP_PATH = os.path.join(BASE_DIR, "dataset_backup.csv")

HOME_BANNER_URL = (
    "https://images.pexels.com/photos/16420457/pexels-photo-16420457/"
    "free-photo-of-a-group-of-students-sitting-in-a-library-and-studying.jpeg"
    "?auto=compress&dpr=1&h=750&w=1260"
)

PAGES = [
    "Home",
    "Dashboard",
    "Student Search",
    "Individual Report",
    "Attendance Analysis",
    "Performance Analysis",
    "ML Prediction",
    "Student Risk Alerts",
    "Student Management",
    "About Project"
]

# =====================================================
# 2. PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #E8EAED;
    color: #303846;
    font-family: Arial, sans-serif !important;
}

html, body, [class*="css"] {
    font-family: Arial, sans-serif !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #203B5C !important;
    font-family: Arial, sans-serif !important;
    font-weight: 600 !important;
}

p, label, li, [data-testid="stMarkdownContainer"] {
    color: #303846;
    font-family: Arial, sans-serif !important;
}

div[data-testid="stAlert"] {
    border-radius: 6px !important;
}

div[data-testid="stAlert"] p {
    color: inherit !important;
}

.stButton > button {
    background-color: #FFFFFF !important;
    color: #203B5C !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    min-height: 40px;
}

.stButton > button p,
.stButton > button span {
    color: #203B5C !important;
}

.stButton > button:hover {
    background-color: #EAF0F7 !important;
    border-color: #7895B5 !important;
}

.stButton > button[kind="primary"] {
    background-color: #203B5C !important;
    color: #FFFFFF !important;
    border-color: #203B5C !important;
}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
}

.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #203B5C !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #303846 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 5px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #303846 !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E1E6ED !important;
    border-radius: 7px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #526174 !important;
}

div[data-testid="stMetricValue"] {
    color: #203B5C !important;
    font-weight: 600 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E1E6ED;
    border-radius: 6px;
}

.stTabs [data-baseweb="tab"] {
    color: #203B5C !important;
    font-weight: 600 !important;
}

details {
    background-color: #FFFFFF !important;
    border: 1px solid #E1E6ED !important;
    border-radius: 6px !important;
}

details summary {
    color: #203B5C !important;
}

.stCaption,
div[data-testid="stCaptionContainer"] {
    color: #526174 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E1E6ED;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100%;
    margin-bottom: 5px;
}

/* Mobile responsive styling */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem;
    }

    h1 {
        font-size: 1.7rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    div[data-testid="stMetric"] {
        padding: 10px;
    }

    .stButton > button {
        min-height: 42px;
        font-size: 13px;
    }
}

@media (max-width: 480px) {
    .block-container {
        padding: 0.7rem 0.5rem;
    }

    div[data-testid="stMetric"] {
        padding: 8px;
    }
}

header[data-testid="stHeader"] {
    height: 2rem !important;
    min-height: 2rem !important;
    background-color: #F7F8FA;
    border-bottom: none;
}

/* Sidebar Toggle Arrow - Pure Black */
header button[data-testid="stSidebarCollapsedControl"],
header button[data-testid="stSidebarCollapseButton"],
header [data-testid="stSidebarCollapsedControl"] button,
header [data-testid="stSidebarCollapseButton"] button {
    color: #000000 !important;
    opacity: 1 !important;
}

header button[data-testid="stSidebarCollapsedControl"] svg,
header button[data-testid="stSidebarCollapseButton"] svg,
header [data-testid="stSidebarCollapsedControl"] svg,
header [data-testid="stSidebarCollapseButton"] svg {
    color: #000000 !important;
    stroke: #000000 !important;
    fill: #000000 !important;
}

header button[data-testid="stSidebarCollapsedControl"] svg path,
header button[data-testid="stSidebarCollapseButton"] svg path {
    stroke: #000000 !important;
    stroke-width: 2.5px !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATA LOADING
# =====================================================

@st.cache_data
def load_data():

    columns = [
        "Student_ID", "Name", "Math",
        "Science", "English", "Attendance"
    ]

    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=columns)

    data = pd.read_csv(DATA_PATH)

    for column in columns:
        if column not in data.columns:
            data[column] = "" if column in [
                "Student_ID", "Name"
            ] else 0

    data = data[columns].copy()

    data["Student_ID"] = data["Student_ID"].astype(str).str.strip()
    data["Name"] = data["Name"].astype(str).str.strip()

    for column in ["Math", "Science", "English", "Attendance"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).fillna(0)

    return data


def save_data(data):

    if not os.path.exists(BACKUP_PATH) and os.path.exists(DATA_PATH):
        shutil.copy2(DATA_PATH, BACKUP_PATH)

    data.to_csv(DATA_PATH, index=False)
    st.cache_data.clear()


df = load_data()


# =====================================================
# 4. COMMON FUNCTIONS
# =====================================================

def show_title(title, description=None):
    st.title(title)

    if description:
        st.write(description)

    st.divider()


def show_info(message):
    st.markdown(
        f"""
        <div style="
            background-color: #EAF1F8;
            color: #203B5C;
            padding: 16px;
            border: 1px solid #CBD8E6;
            border-radius: 6px;
            font-family: Arial, sans-serif;
            font-size: 15px;
            margin: 8px 0;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


def calculate_average(row):
    return (
        float(row["Math"])
        + float(row["Science"])
        + float(row["English"])
    ) / 3


def student_result(row):
    if (
        row["Math"] >= 40
        and row["Science"] >= 40
        and row["English"] >= 40
        and row["Attendance"] >= 75
    ):
        return "Pass"

    return "Needs Improvement"


def student_risk(row):
    average = calculate_average(row)

    if average < 40 or row["Attendance"] < 60:
        return "High Risk"

    if average < 55 or row["Attendance"] < 75:
        return "Medium Risk"

    return "Low Risk"


def find_student(student_id):
    result = df[df["Student_ID"] == str(student_id)]

    if result.empty:
        return None

    return result.iloc[0]


def export_excel(data):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(
            writer,
            index=False,
            sheet_name="Students"
        )

    return output.getvalue()


# =====================================================
# 5. MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None

    return None


model = load_model()


# =====================================================
# 6. LEFT SIDEBAR NAVIGATION
# =====================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

with st.sidebar:

    st.markdown("## Student Analytics Pro")

    st.caption("Academic Performance Monitoring System")

    st.divider()

    st.markdown("**MAIN MENU**")

    for page_name in PAGES:

        if st.button(
            page_name,
            key="sidebar_" + page_name,
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.current_page == page_name
                else "secondary"
            )
        ):
            st.session_state.current_page = page_name
            st.rerun()

    st.divider()

    st.caption("Student Academic Analytics System")

page = st.session_state.current_page


# =====================================================
# 7. HOME PAGE
# =====================================================

if page == "Home":

    intro_col, image_col = st.columns([1.15, 1], gap="large")

    with intro_col:
        st.title("Welcome to Student Analytics Pro")

        st.write(
            "An academic analytics platform designed to understand "
            "student performance, monitor attendance and support "
            "better academic decisions."
        )

    with image_col:
        st.image(
            HOME_BANNER_URL,
            caption=" ",
            use_container_width=True
        )

    st.divider()
    st.subheader("Platform Overview")

    total_students = len(df)

    average_marks = (
        df[["Math", "Science", "English"]].mean().mean()
        if total_students > 0 else 0
    )

    average_attendance = (
        df["Attendance"].mean()
        if total_students > 0 else 0
    )

    at_risk_students = sum(
        student_risk(row) in ["High Risk", "Medium Risk"]
        for _, row in df.iterrows()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total_students)
    c2.metric("Average Marks", f"{average_marks:.2f}")
    c3.metric("Average Attendance", f"{average_attendance:.2f}%")
    c4.metric("At-Risk Students", at_risk_students)

    st.divider()
    st.subheader("Explore the Platform")

    home_pages = [
        ("Dashboard", "View academic statistics and charts."),
        ("Student Search", "Find student records."),
        ("Individual Report", "View an individual student report."),
        ("ML Prediction", "Predict student academic results."),
        ("Student Management", "Add, edit and delete student records."),
        ("Student Risk Alerts", "Identify students who need support.")
    ]

    for start in range(0, len(home_pages), 3):

        columns = st.columns(3)

        for column, item in zip(columns, home_pages[start:start + 3]):

            with column:
                with st.container(border=True):
                    st.subheader(item[0])
                    st.write(item[1])

    st.divider()

    show_info(
        "Select any module from the left sidebar to explore "
        "different sections of the application."
    )


# =====================================================
# 8. SMART DASHBOARD
# =====================================================

elif page == "Dashboard":

    show_title(
        "Smart Dashboard",
        "Overview of student academic performance and attendance."
    )

    if df.empty:
        show_info("No student records are available.")

    else:

        filtered_df = df.copy()

        col1, col2 = st.columns(2)

        with col1:
            attendance_filter = st.selectbox(
                "Attendance Filter",
                ["All", "Below 75%", "75% and Above"]
            )

        with col2:
            subject_filter = st.selectbox(
                "Performance Subject",
                ["All Subjects", "Math", "Science", "English"]
            )

        if attendance_filter == "Below 75%":
            filtered_df = filtered_df[
                filtered_df["Attendance"] < 75
            ]

        elif attendance_filter == "75% and Above":
            filtered_df = filtered_df[
                filtered_df["Attendance"] >= 75
            ]

        total = len(filtered_df)

        average = (
            filtered_df[["Math", "Science", "English"]].mean().mean()
            if total else 0
        )

        attendance = (
            filtered_df["Attendance"].mean()
            if total else 0
        )

        passed = sum(
            student_result(row) == "Pass"
            for _, row in filtered_df.iterrows()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Students", total)
        c2.metric("Average Marks", f"{average:.2f}")
        c3.metric("Average Attendance", f"{attendance:.2f}%")
        c4.metric("Students Passing", passed)

        st.divider()
        st.subheader("Student Records")

        display_df = filtered_df.copy()

        display_df["Average Marks"] = (
            display_df.apply(calculate_average, axis=1).round(2)
        )

        display_df["Result"] = display_df.apply(
            student_result, axis=1
        )

        st.dataframe(display_df, use_container_width=True)

        st.divider()

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:

            st.subheader("Subject-wise Average Marks")

            fig, ax = plt.subplots(figsize=(6, 4))

            if subject_filter == "All Subjects":

                averages = filtered_df[
                    ["Math", "Science", "English"]
                ].mean()

                averages.plot(
                    kind="bar",
                    ax=ax,
                    color="#2878B5"
                )

                ax.set_ylabel("Average Marks")
                ax.set_ylim(0, 100)
                ax.set_xlabel("Subject")
                ax.tick_params(axis="x", rotation=0)

            else:

                ax.hist(
                    filtered_df[subject_filter],
                    bins=10,
                    color="#2878B5",
                    edgecolor="black"
                )

                ax.set_xlabel("Marks")
                ax.set_ylabel("Number of Students")
                ax.set_title(f"{subject_filter} Marks Distribution")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with chart2:

            st.subheader("Student-wise Attendance")

            fig, ax = plt.subplots(figsize=(6, 4))

            if not filtered_df.empty:

                attendance_data = filtered_df.sort_values("Student_ID")

                bars = ax.bar(
                    attendance_data["Student_ID"].astype(str),
                    attendance_data["Attendance"],
                    color="#2878B5",
                    edgecolor="black",
                    width=0.65
                )

                ax.set_ylim(0, 110)
                ax.set_xlabel("Student ID")
                ax.set_ylabel("Attendance (%)")

                ax.set_title(
                    f"Attendance of {len(attendance_data)} Students"
                )

                ax.tick_params(axis="x", rotation=45)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 1,
                        f"{height:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

            else:
                ax.text(
                    0.5, 0.5,
                    "No students in this filter",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )
                ax.set_axis_off()

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.download_button(
            "Download Dashboard Data",
            data=export_excel(display_df),
            file_name="student_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =====================================================
# 9. STUDENT SEARCH
# =====================================================

elif page == "Student Search":

    show_title(
        "Student Search",
        "Search student records using Student ID or Name."
    )

    if df.empty:
        show_info("No student records are available.")

    else:

        search_type = st.radio(
            "Search By",
            ["Student ID", "Name"],
            horizontal=True
        )

        search_text = st.text_input("Enter search value")

        if search_text.strip():

            if search_type == "Student ID":
                result = df[
                    df["Student_ID"].str.contains(
                        search_text.strip(),
                        case=False,
                        na=False
                    )
                ]

            else:
                result = df[
                    df["Name"].str.contains(
                        search_text.strip(),
                        case=False,
                        na=False
                    )
                ]

            if result.empty:
                show_info("No matching student found.")

            else:

                st.success(
                    f"{len(result)} student record(s) found."
                )

                st.dataframe(result, use_container_width=True)

                selected_id = st.selectbox(
                    "Select Student",
                    result["Student_ID"].tolist()
                )

                if st.button("View Student Report"):
                    st.session_state.report_student_id = selected_id
                    st.session_state.current_page = "Individual Report"
                    st.rerun()


# =====================================================
# 10. INDIVIDUAL STUDENT REPORT
# =====================================================

elif page == "Individual Report":

    show_title(
        "Individual Student Report",
        "View complete academic details of an individual student."
    )

    if df.empty:
        show_info("No student records are available.")

    else:

        options = df["Student_ID"].tolist()

        default_id = st.session_state.get(
            "report_student_id",
            options[0]
        )

        if default_id not in options:
            default_id = options[0]

        selected_id = st.selectbox(
            "Select Student ID",
            options,
            index=options.index(default_id)
        )

        student = find_student(selected_id)

        if student is not None:

            average = calculate_average(student)

            st.divider()
            st.subheader(student["Name"])

            c1, c2, c3 = st.columns(3)

            c1.metric("Student ID", student["Student_ID"])
            c2.metric("Average Marks", f"{average:.2f}")
            c3.metric("Attendance", f"{student['Attendance']:.2f}%")

            st.divider()
            st.subheader("Subject Performance")

            marks_df = pd.DataFrame({
                "Subject": ["Math", "Science", "English"],
                "Marks": [
                    student["Math"],
                    student["Science"],
                    student["English"]
                ]
            })

            st.dataframe(marks_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.bar(
                marks_df["Subject"],
                marks_df["Marks"],
                color="#2878B5"
            )

            ax.set_ylim(0, 100)
            ax.set_ylabel("Marks")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.subheader("Academic Summary")

            st.write("**Result:**", student_result(student))
            st.write("**Risk Category:**", student_risk(student))

            report = pd.DataFrame([{
                "Student_ID": student["Student_ID"],
                "Name": student["Name"],
                "Math": student["Math"],
                "Science": student["Science"],
                "English": student["English"],
                "Average Marks": round(average, 2),
                "Attendance": student["Attendance"],
                "Result": student_result(student),
                "Risk": student_risk(student)
            }])

            st.download_button(
                "Download Student Report",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name=f"student_{selected_id}_report.csv",
                mime="text/csv"
            )


# =====================================================
# 11. ATTENDANCE ANALYSIS
# =====================================================

elif page == "Attendance Analysis":

    show_title(
        "Attendance Analysis",
        "Analyze student attendance and identify attendance concerns."
    )

    if df.empty:

        show_info("No student records are available.")

    else:

        average_attendance = df["Attendance"].mean()

        st.write("Total Students:", len(df))
        st.write("Attendance Records:", df["Attendance"].count())

        below_75 = df[df["Attendance"] < 75].copy()
        above_75 = df[df["Attendance"] >= 75].copy()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Attendance",
            f"{average_attendance:.2f}%"
        )

        c2.metric("Below 75%", len(below_75))
        c3.metric("75% and Above", len(above_75))

        st.divider()
        st.subheader("Student-wise Attendance")

        attendance_data = df.sort_values("Student_ID")

        fig, ax = plt.subplots(figsize=(9, 4.5))

        bars = ax.bar(
            attendance_data["Student_ID"].astype(str),
            attendance_data["Attendance"],
            color="#2878B5",
            edgecolor="black",
            width=0.65
        )

        ax.set_ylim(0, 110)
        ax.set_xlabel("Student ID")
        ax.set_ylabel("Attendance (%)")

        ax.set_title(
            f"Attendance of {len(attendance_data)} Students"
        )

        ax.tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()
        st.subheader("Students with Low Attendance")

        if below_75.empty:

            st.success(
                "All students have attendance of 75% or above."
            )

        else:

            st.dataframe(
                below_75.sort_values("Attendance"),
                use_container_width=True
            )

            st.download_button(
                "Download Low Attendance Report",
                data=below_75.to_csv(index=False).encode("utf-8"),
                file_name="low_attendance_report.csv",
                mime="text/csv"
            )


# =====================================================
# 12. PERFORMANCE ANALYSIS
# =====================================================

elif page == "Performance Analysis":

    show_title(
        "Performance Analysis",
        "Understand subject-wise performance and student results."
    )

    if df.empty:

        show_info("No student records are available.")

    else:

        averages = df[
            ["Math", "Science", "English"]
        ].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Math Average", f"{averages['Math']:.2f}")
        c2.metric("Science Average", f"{averages['Science']:.2f}")
        c3.metric("English Average", f"{averages['English']:.2f}")

        st.divider()

        chart1, chart2 = st.columns(2, gap="large")

        with chart1:

            st.subheader("Subject-wise Performance")

            fig, ax = plt.subplots(figsize=(6, 4))

            averages.plot(
                kind="bar",
                ax=ax,
                color="#2878B5"
            )

            ax.set_ylabel("Average Marks")
            ax.set_ylim(0, 100)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with chart2:

            st.subheader("Student Average Marks")

            average_data = df.copy()

            average_data["Average"] = (
                average_data.apply(calculate_average, axis=1)
            )

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.hist(
                average_data["Average"],
                bins=10,
                color="#2878B5",
                edgecolor="black"
            )

            ax.set_xlabel("Average Marks")
            ax.set_ylabel("Number of Students")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()
        st.subheader("Top Performing Students")

        top_students = df.copy()

        top_students["Average Marks"] = (
            top_students.apply(calculate_average, axis=1)
        )

        top_students = top_students.sort_values(
            "Average Marks",
            ascending=False
        ).head(5)

        st.dataframe(top_students, use_container_width=True)


# =====================================================
# 13. MACHINE LEARNING PREDICTION
# =====================================================

elif page == "ML Prediction":

    show_title(
        "Machine Learning Prediction",
        "Predict student academic results using the trained model."
    )

    if model is None:

        st.error(
            "The trained model could not be loaded. "
            "Please check student_model.pkl."
        )

    else:

        st.write(
            "Enter the student's academic information "
            "to generate a prediction."
        )

        with st.form("prediction_form"):

            c1, c2 = st.columns(2)

            with c1:

                studytime = st.number_input(
                    "Study Time (1-4)",
                    min_value=1,
                    max_value=4,
                    value=2
                )

                failures = st.number_input(
                    "Past Failures",
                    min_value=0,
                    max_value=4,
                    value=0
                )

                absences = st.number_input(
                    "Absences",
                    min_value=0,
                    max_value=100,
                    value=2
                )

                age = st.number_input(
                    "Age",
                    min_value=15,
                    max_value=25,
                    value=18
                )

            with c2:

                g1 = st.number_input(
                    "First Period Grade (G1)",
                    min_value=0,
                    max_value=20,
                    value=10
                )

                g2 = st.number_input(
                    "Second Period Grade (G2)",
                    min_value=0,
                    max_value=20,
                    value=10
                )

                health = st.number_input(
                    "Health (1-5)",
                    min_value=1,
                    max_value=5,
                    value=3
                )

                freetime = st.number_input(
                    "Free Time (1-5)",
                    min_value=1,
                    max_value=5,
                    value=3
                )

            submitted = st.form_submit_button(
                "Predict Student Result",
                use_container_width=True
            )

        if submitted:

            features = [[
                studytime,
                failures,
                absences,
                g1,
                g2,
                age,
                health,
                freetime
            ]]

            try:

                prediction = model.predict(features)[0]

                st.divider()
                st.subheader("Prediction Result")

                if str(prediction).lower() in ["pass", "1", "true"]:
                    st.success("Predicted Result: Pass")
                else:
                    st.warning("Predicted Result: Fail")

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(features)[0]

                    st.subheader("Prediction Confidence")

                    probability_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Probability (%)": probabilities * 100
                    })

                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

            except Exception as error:

                st.error(f"Prediction failed: {error}")

        if hasattr(model, "feature_importances_"):

            st.divider()
            st.subheader("Model Feature Importance")

            feature_names = [
                "Study Time",
                "Failures",
                "Absences",
                "G1",
                "G2",
                "Age",
                "Health",
                "Free Time"
            ]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance",
                ascending=False
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"],
                color="#2878B5"
            )

            ax.set_xlabel("Importance")
            ax.invert_yaxis()

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# =====================================================
# 14. STUDENT RISK ALERTS
# =====================================================

elif page == "Student Risk Alerts":

    show_title(
        "Student Risk Alerts",
        "Identify students who may need additional academic support."
    )

    if df.empty:

        show_info("No student records are available.")

    else:

        risk_df = df.copy()

        risk_df["Average Marks"] = (
            risk_df.apply(calculate_average, axis=1)
        )

        risk_df["Risk Category"] = (
            risk_df.apply(student_risk, axis=1)
        )

        high_risk = risk_df[
            risk_df["Risk Category"] == "High Risk"
        ]

        medium_risk = risk_df[
            risk_df["Risk Category"] == "Medium Risk"
        ]

        low_risk = risk_df[
            risk_df["Risk Category"] == "Low Risk"
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric("High Risk", len(high_risk))
        c2.metric("Medium Risk", len(medium_risk))
        c3.metric("Low Risk", len(low_risk))

        st.divider()
        st.subheader("Risk Distribution")

        risk_counts = risk_df["Risk Category"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        risk_counts.plot(
            kind="bar",
            ax=ax,
            color="#2878B5"
        )

        ax.set_ylabel("Number of Students")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Student Risk Details")

        selected_risk = st.selectbox(
            "Filter by Risk Category",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )

        if selected_risk != "All":

            display_risk = risk_df[
                risk_df["Risk Category"] == selected_risk
            ]

        else:

            display_risk = risk_df

        st.dataframe(display_risk, use_container_width=True)

        st.subheader("Suggested Academic Support")

        st.write(
            "High Risk: Individual attention, academic counselling "
            "and regular monitoring."
        )

        st.write(
            "Medium Risk: Additional practice and attendance monitoring."
        )

        st.write(
            "Low Risk: Continue regular academic progress monitoring."
        )

        st.download_button(
            "Download Risk Report",
            data=risk_df.to_csv(index=False).encode("utf-8"),
            file_name="student_risk_report.csv",
            mime="text/csv"
        )


# =====================================================
# 15. STUDENT MANAGEMENT
# =====================================================

elif page == "Student Management":

    show_title(
        "Student Management",
        "Add new students, update records or delete student records."
    )

    tab1, tab2, tab3 = st.tabs([
        "Add Student",
        "Edit Student",
        "Delete Student"
    ])

    with tab1:

        st.subheader("Add New Student")

        with st.form("add_student_form"):

            student_id = st.text_input("Student ID")
            student_name = st.text_input("Student Name")

            c1, c2 = st.columns(2)

            with c1:

                math = st.number_input(
                    "Math Marks",
                    min_value=0,
                    max_value=100,
                    value=0
                )

                science = st.number_input(
                    "Science Marks",
                    min_value=0,
                    max_value=100,
                    value=0
                )

            with c2:

                english = st.number_input(
                    "English Marks",
                    min_value=0,
                    max_value=100,
                    value=0
                )

                attendance = st.number_input(
                    "Attendance (%)",
                    min_value=0,
                    max_value=100,
                    value=0
                )

            add_submitted = st.form_submit_button(
                "Add Student",
                use_container_width=True
            )

        if add_submitted:

            student_id = student_id.strip()
            student_name = student_name.strip()

            if not student_id or not student_name:

                st.error("Please enter Student ID and Student Name.")

            elif student_id in df["Student_ID"].values:

                st.error("This Student ID already exists.")

            else:

                new_student = pd.DataFrame([{
                    "Student_ID": student_id,
                    "Name": student_name,
                    "Math": math,
                    "Science": science,
                    "English": english,
                    "Attendance": attendance
                }])

                updated_df = pd.concat(
                    [df, new_student],
                    ignore_index=True
                )

                save_data(updated_df)

                st.success("Student added successfully.")
                st.rerun()

    with tab2:

        st.subheader("Edit Student Details")

        if df.empty:

            show_info("No student records available.")

        else:

            edit_id = st.selectbox(
                "Select Student ID to Edit",
                df["Student_ID"].tolist(),
                key="edit_student_id"
            )

            selected_student = find_student(edit_id)

            if selected_student is not None:

                with st.form("edit_student_form"):

                    edit_name = st.text_input(
                        "Student Name",
                        value=selected_student["Name"]
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        edit_math = st.number_input(
                            "Math Marks",
                            min_value=0,
                            max_value=100,
                            value=int(selected_student["Math"]),
                            key="edit_math"
                        )

                        edit_science = st.number_input(
                            "Science Marks",
                            min_value=0,
                            max_value=100,
                            value=int(selected_student["Science"]),
                            key="edit_science"
                        )

                    with c2:

                        edit_english = st.number_input(
                            "English Marks",
                            min_value=0,
                            max_value=100,
                            value=int(selected_student["English"]),
                            key="edit_english"
                        )

                        edit_attendance = st.number_input(
                            "Attendance (%)",
                            min_value=0,
                            max_value=100,
                            value=int(selected_student["Attendance"]),
                            key="edit_attendance"
                        )

                    edit_submitted = st.form_submit_button(
                        "Save Changes",
                        use_container_width=True
                    )

                if edit_submitted:

                    if not edit_name.strip():

                        st.error("Student name cannot be empty.")

                    else:

                        updated_df = df.copy()

                        index = updated_df.index[
                            updated_df["Student_ID"] == edit_id
                        ][0]

                        updated_df.loc[index, "Name"] = edit_name.strip()
                        updated_df.loc[index, "Math"] = edit_math
                        updated_df.loc[index, "Science"] = edit_science
                        updated_df.loc[index, "English"] = edit_english
                        updated_df.loc[index, "Attendance"] = edit_attendance

                        save_data(updated_df)

                        st.success(
                            "Student details updated successfully."
                        )

                        st.rerun()

    with tab3:

        st.subheader("Delete Student Record")

        if df.empty:

            show_info("No student records available.")

        else:

            delete_id = st.selectbox(
                "Select Student ID to Delete",
                df["Student_ID"].tolist(),
                key="delete_student_id"
            )

            delete_student = find_student(delete_id)

            if delete_student is not None:

                st.write("**Student Name:**", delete_student["Name"])
                st.write("**Student ID:**", delete_student["Student_ID"])

                confirm_delete = st.checkbox(
                    "I confirm that I want to delete this student record."
                )

                if st.button(
                    "Delete Student",
                    type="primary",
                    use_container_width=True
                ):

                    if not confirm_delete:

                        st.error("Please confirm before deleting.")

                    else:

                        updated_df = df[
                            df["Student_ID"] != delete_id
                        ].copy()

                        save_data(updated_df)

                        st.success(
                            "Student record deleted successfully."
                        )

                        st.rerun()

    st.divider()
    st.subheader("Current Student Records")

    st.dataframe(df, use_container_width=True)

    st.caption(
        "Changes are saved to dataset.csv on the machine running "
        "the application. A backup is created before the first change."
    )


# =====================================================
# 16. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    show_title(
        "About Student Analytics Pro",
        "Project overview, technology stack and machine learning details."
    )

    st.subheader("Project Introduction")

    st.write(
        "Student Analytics Pro is an academic analytics application "
        "developed to analyze student performance, monitor attendance "
        "and identify students who may need academic support."
    )

    st.divider()
    st.subheader("Project Objectives")

    objectives = [
        "Monitor student academic performance.",
        "Analyze attendance patterns.",
        "Generate individual student reports.",
        "Predict academic outcomes using machine learning.",
        "Identify students who may be at academic risk.",
        "Manage student records."
    ]

    for objective in objectives:
        st.write("- " + objective)

    st.divider()
    st.subheader("Technology Stack")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "Matplotlib",
            "Scikit-learn",
            "Joblib"
        ],
        "Purpose": [
            "Application development",
            "Web application interface",
            "Data processing",
            "Data visualization",
            "Machine learning",
            "Model loading"
        ]
    })

    st.dataframe(technologies, use_container_width=True)

    st.divider()
    st.subheader("Machine Learning Model")

    st.write(
        "The application uses a trained Decision Tree classification "
        "model to predict student academic results."
    )

    st.write(
        "The prediction inputs include study time, past failures, "
        "absences, previous grades, age, health and free time."
    )

    st.write(
        "The model predicts whether a student is likely to pass or fail "
        "based on patterns learned during training."
    )

    st.divider()
    st.subheader("Application Modules")

    for module in PAGES:
        st.write("- " + module)

    st.divider()

    st.caption(
        "Student Analytics Pro | Academic Performance Analysis Project"
    )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.caption(
    "Student Analytics Pro | Student Academic Analytics System"
)
