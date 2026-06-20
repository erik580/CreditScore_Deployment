import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="wide"
)

# Load model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def main():

    st.title("Credit Score Prediction")

    st.markdown(
        """
        Predict customer credit score using financial,
        credit profile, and behavioural information.
        """
    )

    st.divider()

    # Input Tab
    tab1, tab2, tab3 = st.tabs(
        [
            "Personal & Income",
            "Credit Profile",
            "Behaviour & Debt"
        ]
    )


    # ======= Tab 1 - Personal Income =======
    with tab1:

        occupation = st.selectbox(
            "Occupation",
            [
                "Lawyer",
                "Mechanic",
                "Teacher",
                "Developer",
                "Journalist",
                "Scientist",
                "Accountant",
                "Media_Manager",
                "Architect"
            ]
        )

        annual_income = st.text_input(
            "Annual Income",
            value="50000"
        )

        monthly_salary = st.number_input(
            "Monthly Inhand Salary",
            min_value=0.0,
            value=4000.0
        )

        num_bank_accounts = st.number_input(
            "Number of Bank Accounts",
            min_value=0,
            value=4
        )

        num_credit_card = st.number_input(
            "Number of Credit Cards",
            min_value=0,
            value=3
        )

        amount_invested_monthly = st.text_input(
            "Amount Invested Monthly",
            value="500"
        )

        monthly_balance = st.number_input(
            "Monthly Balance",
            value=1000.0
        )


    # ======= Tab 2 - Credit Profile =======
    with tab2:

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0,
            max_value=100,
            value=5
        )

        num_of_loan = st.text_input(
            "Number of Loans",
            value="2"
        )

        selected_loans = st.multiselect(
            "Type of Loan",
            [
                "Not Specified",
                "Credit-Builder Loan",
                "Personal Loan",
                "Debt Consolidation Loan",
                "Student Loan",
                "Payday Loan",
                "Home Equity Loan",
                "Mortgage Loan",
                "Auto Loan"
            ],
            default=["Not Specified"]
        )

        # format mengikuti dataset asli
        if len(selected_loans) == 1:
            type_of_loan = selected_loans[0]

        elif len(selected_loans) > 1:
            type_of_loan = ", ".join(selected_loans[:-1])
            type_of_loan += f", and {selected_loans[-1]}"

        else:
            type_of_loan = "Not Specified"

        delay_due_date = st.number_input(
            "Delay From Due Date",
            min_value=0,
            value=5
        )

        num_delayed_payment = st.text_input(
            "Number of Delayed Payments",
            value="3"
        )

        changed_credit_limit = st.text_input(
            "Changed Credit Limit",
            value="5.5"
        )

        credit_inquiries = st.number_input(
            "Credit Inquiries",
            min_value=0.0,
            value=3.0
        )


    # ======= Tab 3 - Behaviour & Debt =======
    with tab3:

        credit_mix = st.selectbox(
            "Credit Mix",
            ["Bad", "Standard", "Good"]
        )

        outstanding_debt = st.text_input(
            "Outstanding Debt",
            value="1000"
        )

        credit_utilization = st.number_input(
            "Credit Utilization Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0
        )

        st.markdown("Credit History Age")

        col_year, col_month = st.columns([2, 1])

        with col_year:
            credit_history_year = st.number_input(
                "Years",
                min_value=0,
                max_value=100,
                value=15
            )

        with col_month:
            credit_history_month = st.number_input(
                "Months",
                min_value=0,
                max_value=11,
                value=6
            )

        credit_history = (
            f"{credit_history_year} Years and " f"{credit_history_month} Months"
        )

        payment_min_amount = st.selectbox(
            "Payment of Min Amount",
            ["Yes", "No"]
        )

        total_emi = st.number_input(
            "Total EMI per Month",
            min_value=0.0,
            value=100.0
        )

        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments"
            ]
        )

    st.divider()

    # Predict Button
    left, center, right = st.columns([3, 1, 3])

    with center:
        predict_button = st.button(
            "Predict Credit Score",
            use_container_width=True
        )


    # Predict
    if predict_button:

        data = {
            "Occupation": occupation,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_salary,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,
            "Type_of_Loan": type_of_loan,
            "Delay_from_due_date": delay_due_date,
            "Num_of_Delayed_Payment": num_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": credit_inquiries,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization,
            "Credit_History_Age": credit_history,
            "Payment_of_Min_Amount": payment_min_amount,
            "Total_EMI_per_month": total_emi,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": monthly_balance
        }

        df = pd.DataFrame([data])

        # Buat tampilin input summary
        st.divider()

        st.subheader("📝 Input Summary")

        col1, col2, col3 = st.columns(3)
        
        with col1:

            st.markdown("### 👤 Personal & Income")

            summary_personal = pd.DataFrame({
                "Feature": [
                    "Occupation",
                    "Annual Income",
                    "Monthly Salary",
                    "Bank Accounts",
                    "Credit Cards",
                    "Invested Monthly",
                    "Monthly Balance"
                ],
                "Value": [
                    occupation,
                    annual_income,
                    monthly_salary,
                    num_bank_accounts,
                    num_credit_card,
                    amount_invested_monthly,
                    monthly_balance
                ]
            })

            st.dataframe(
                summary_personal,
                hide_index=True,
                use_container_width=True
            )

        
        with col2:

            st.markdown("### 🏦 Credit Profile")

            summary_credit = pd.DataFrame({
                "Feature": [
                    "Interest Rate",
                    "Number of Loans",
                    "Type of Loan",
                    "Delay Due Date",
                    "Delayed Payments",
                    "Changed Credit Limit",
                    "Credit Inquiries"
                ],
                "Value": [
                    interest_rate,
                    num_of_loan,
                    type_of_loan,
                    delay_due_date,
                    num_delayed_payment,
                    changed_credit_limit,
                    credit_inquiries
                ]
            })

            st.dataframe(
                summary_credit,
                hide_index=True,
                use_container_width=True
            )

        
        with col3:

            st.markdown("### 📈 Behaviour & Debt")
        
            summary_behaviour = pd.DataFrame({
                "Feature": [
                    "Credit Mix",
                    "Outstanding Debt",
                    "Credit Utilization",
                    "Credit History",
                    "Min Amount Payment",
                    "Total EMI",
                    "Payment Behaviour"
                ],
                "Value": [
                    credit_mix,
                    outstanding_debt,
                    f"{credit_utilization}%",
                    credit_history,
                    payment_min_amount,
                    total_emi,
                    payment_behaviour
                ]
            })
        
            st.dataframe(
                summary_behaviour,
                hide_index=True,
                use_container_width=True
            )

        # Melakukan prediksi
        prediction = model.predict(df)[0]

        mapping = {
            0: "Poor",
            1: "Standard",
            2: "Good"
        }

        predicted_label = mapping[prediction]

        st.divider()

        st.subheader("📊 Prediction Result")

        # 3 case agar output berwarna berdasarkan tingkat credit score nya
        if predicted_label == "Good":
            st.success(f"Predicted Credit Score: {predicted_label}")

        elif predicted_label == "Standard":
            st.warning(f"Predicted Credit Score: {predicted_label}")

        else:
            st.error(f"Predicted Credit Score: {predicted_label}")


if __name__ == "__main__":
    main()