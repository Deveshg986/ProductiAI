import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="ProductiAI", layout="wide", page_icon="Logo.png")

#Read data from the data.csv
df = pd.read_csv("data.csv")

# x= features(droping the column Productivity coz we need the feature which affect the productivity)
X = df.drop("Productivity", axis=1)
y = df["Productivity"]

#Ml Model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X, y)

#Frontend 
st.title("ProductiAI")
tab1, tab2, tab3 = st.tabs(["Home", "Analyze", "About"])

# Home Page
with tab1:
    st.title("Productivity Analyzer")

    st.markdown("""
    ### Welcome
    This Tool help you to understand how your daily habits impact productivity.
    it use **Machine Learning** and **real-world logic** to provide accurate and meaningfull insight
    """)
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Analyze")
        st.write("Track daily habbits like sleep work and stres")
    with col2:
        st.subheader("Predict")
        st.write("Get an AI-based productivity score instantly.")
    with col3:
        st.subheader("Improve")
        st.write("Receive smart suggestions to boost productivity.")
    
    st.divider()

    st.subheader("Why You Should Use this Tool ?")

    st.markdown("""
    - Improve your daily productivity  
    - Understand the impact of your habits  
    - Make smarter lifestyle decisions  
    """)
with tab2:
    st.title("Analyze Your Productivity")

    col1, col2 = st.columns(2)

    #Inputs For Habits
    with col1:
        st.subheader("Input Your Habits")

        sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        screen = st.slider("Screen Time (hrs)", 0.0, 12.0, 5.0)
        work = st.slider("Work Hours", 0.0, 12.0, 8.0)
        exercise = st.slider("Exercise (hrs)", 0.0, 3.0, 1.0)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)

        analyze = st.button("Analyze")

    #Results
    with col2:
        st.subheader("Results")

        if analyze:
            user_data = [[sleep, screen, work, exercise, stress]]

            prediction = model.predict(user_data)[0]

            if stress > 7:
                prediction -= 10

            if work < 5:
                prediction -= 10

            if exercise == 0:
                prediction -= 5

            if sleep > 10:
                prediction -= 5

            if work > 8 and stress < 3:
                prediction += 5

            if exercise > 1:
                prediction += 3

            # range of the Prediction 0-100
            prediction = max(0, min(100, prediction))

            st.metric("Productivity Score", round(prediction, 2))

            if prediction > 70:
                st.success("High Productivity")
            elif prediction > 50:
                st.warning("Moderate Productivity")
            else:
                st.error("Low Productivity")

            st.markdown("### Suggestions")

            suggestion_given = False

            if sleep < 6:
                st.info("Increase your sleep.")
                suggestion_given = True

            if screen > 5:
                st.warning("Reduce screen time.")
                suggestion_given = True

            if stress > 7:
                st.error("Manage stress.")
                suggestion_given = True

            if exercise < 0.5:
                st.info("Add exercise.")
                suggestion_given = True

            if not suggestion_given:
                st.success("Your habits are great!")

# About Page
with tab3:
    st.title("About This App")

    st.markdown("""
    This project is built using:

    - Machine Learning (Random Forest)
    - Pandas for data handling
    - Streamlit for UI

    ### Purpose
    To help users understand how lifestyle affects productivity.
        ---

    ### How It Works

    - The system takes user inputs such as sleep, work hours, stress level, etc.  
    - A trained **Random Forest model** predicts the productivity score.  
    - A **calibration layer** adjusts predictions to match real-world behavior.  

    ---

    ### Purpose of the Project

    The goal of this project is to:
    - Help users understand the impact of daily habits on productivity  
    - Provide actionable insights to improve lifestyle  
    - Demonstrate the use of Machine Learning in real-world scenarios  

    ---

    ### Future Enhancements

    - User login and data storage  
    - Productivity tracking over time  
    - Advanced AI-based recommendations  
    - Data visualization (graphs & trends)  

    ---
    """)
