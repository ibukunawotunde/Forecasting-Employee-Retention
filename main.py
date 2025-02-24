# --- DEI Metrics Dashboard ---
if page == "DEI Metrics":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("🌍 Diversity, Equity & Inclusion (DEI) Metrics")
    st.write("Analyze attrition rates by gender, department, job role, salary level, and work-life balance to identify disparities.")

    # Ensure required columns exist
    required_columns = ["ATTRITION", "GENDER", "DEPARTMENT", "JOB ROLE", "MARITAL STATUS", 
                        "DAILY RATE", "HOURLY RATE", "MONTHLY INCOME", "OVERTIME", 
                        "WORK LIFE BALANCE", "YEARS SINCE LAST PROMOTION"]
    
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing Columns: {', '.join(missing_cols)}. Please ensure DEI data is available.")
    else:
        # Convert categorical columns
        df["GENDER"] = df["GENDER"].map({1: "Female", 2: "Male"})
        df["DEPARTMENT"] = df["DEPARTMENT"].map({1: "HR", 2: "R&D", 3: "Sales"})
        df["MARITAL STATUS"] = df["MARITAL STATUS"].map({1: "Divorced", 2: "Married", 3: "Single"})
        df["OVERTIME"] = df["OVERTIME"].map({1: "No", 2: "Yes"})

        # Attrition Rate by Gender
        st.subheader("📊 Attrition Rate by Gender")
        gender_attrition = df.groupby("GENDER")["ATTRITION"].mean().reset_index()
        gender_attrition.columns = ["Gender", "Attrition Rate"]
        fig1 = px.bar(gender_attrition, x="Gender", y="Attrition Rate", color="Gender", 
                      title="Attrition Rate by Gender", labels={"Attrition Rate": "Attrition %"})
        st.plotly_chart(fig1, use_container_width=True)

        # Attrition by Department
        st.subheader("🏢 Attrition by Department")
        dept_attrition = df.groupby("DEPARTMENT")["ATTRITION"].mean().reset_index()
        dept_attrition.columns = ["Department", "Attrition Rate"]
        fig2 = px.bar(dept_attrition, x="Department", y="Attrition Rate", color="Department", 
                      title="Attrition Rate by Department", labels={"Attrition Rate": "Attrition %"})
        st.plotly_chart(fig2, use_container_width=True)

        # Attrition by Job Role
        st.subheader("🔬 Attrition by Job Role")
        job_attrition = df.groupby("JOB ROLE")["ATTRITION"].mean().reset_index()
        job_attrition.columns = ["Job Role", "Attrition Rate"]
        fig3 = px.bar(job_attrition, x="Job Role", y="Attrition Rate", color="Job Role", 
                      title="Attrition Rate by Job Role", labels={"Attrition Rate": "Attrition %"})
        st.plotly_chart(fig3, use_container_width=True)

        # Attrition by Salary Level (Daily Rate)
        st.subheader("💰 Attrition by Salary Level")
        fig4 = px.box(df, x="ATTRITION", y="DAILY RATE", color="ATTRITION",
                      title="Salary Distribution Among Employees Who Left vs Stayed",
                      labels={"DAILY RATE": "Salary Level", "ATTRITION": "Attrition"})
        st.plotly_chart(fig4, use_container_width=True)

        # Attrition by Work-Life Balance & Overtime
        st.subheader("⚖️ Work-Life Balance & Overtime Impact on Attrition")
        work_life = df.groupby(["WORK LIFE BALANCE", "OVERTIME"])["ATTRITION"].mean().reset_index()
        work_life.columns = ["Work-Life Balance", "Overtime", "Attrition Rate"]
        fig5 = px.bar(work_life, x="Work-Life Balance", y="Attrition Rate", color="Overtime",
                      title="Work-Life Balance & Overtime Impact on Attrition",
                      barmode="group", labels={"Attrition Rate": "Attrition %"})
        st.plotly_chart(fig5, use_container_width=True)

        # Attrition by Promotion History
        st.subheader("📈 Promotion & Career Growth Impact on Attrition")
        promo_attrition = df.groupby("YEARS SINCE LAST PROMOTION")["ATTRITION"].mean().reset_index()
        promo_attrition.columns = ["Years Since Last Promotion", "Attrition Rate"]
        fig6 = px.line(promo_attrition, x="Years Since Last Promotion", y="Attrition Rate", 
                       title="Attrition Rate by Years Since Last Promotion",
                       labels={"Attrition Rate": "Attrition %"})
        st.plotly_chart(fig6, use_container_width=True)

        # Intersectional Analysis: Gender + Salary + Attrition
        st.subheader("🔥 Intersectional Analysis: Gender & Salary Impact")
        heatmap_data = df.groupby(["GENDER", "SALARY"])["ATTRITION"].mean().unstack()
        fig7, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".2%")
        st.pyplot(fig7)

    st.markdown('</div>', unsafe_allow_html=True)
