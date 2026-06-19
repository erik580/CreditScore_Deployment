from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class CreditScoreFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        # =========================
        # CLEANING
        # =========================

        uninformative = [
            "Unnamed: 0",
            "ID",
            "Customer_ID",
            "Month",
            "Name",
            "Age",
            "SSN"
        ]

        df = df.drop(
            columns=uninformative,
            errors="ignore"
        )

        df["Occupation"] = df["Occupation"].replace(
            "_______",
            np.nan
        )

        df["Credit_Mix"] = df["Credit_Mix"].replace(
            "_",
            np.nan
        )

        df["Payment_of_Min_Amount"] = (
            df["Payment_of_Min_Amount"]
            .replace("NM", np.nan)
        )

        df["Payment_Behaviour"] = (
            df["Payment_Behaviour"]
            .replace("!@9#%8", np.nan)
        )

        df["Amount_invested_monthly"] = (
            df["Amount_invested_monthly"]
            .replace("__10000__", np.nan)
        )

        df["Monthly_Balance"] = (
            df["Monthly_Balance"]
            .replace(
                "__-333333333333333333333333333__",
                np.nan
            )
        )

        # =========================
        # NUMERIC CONVERSION
        # =========================

        numeric_object_cols = [
            'Annual_Income',
            'Num_of_Loan',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Outstanding_Debt',
            'Amount_invested_monthly'
        ]

        for col in numeric_object_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace('_', '', regex=False)
            )

        df["Changed_Credit_Limit"] = (
            df["Changed_Credit_Limit"]
            .replace(["", "_"], np.nan)
        )

        num_convert = [
            'Annual_Income',
            'Num_of_Loan',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Outstanding_Debt',
            'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        for col in num_convert:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
        # =========================
        # OUTLIER CAPPING
        # =========================
        
        df = self.cap_outliers(df)

        # =========================
        # TYPE OF LOAN
        # =========================

        df["Type_of_Loan"] = (
            df["Type_of_Loan"]
            .fillna("Not Specified")
        )

        main_loans = [
            "Not Specified",
            "Credit-Builder Loan",
            "Personal Loan",
            "Debt Consolidation Loan",
            "Student Loan",
            "Payday Loan",
            "Home Equity Loan",
            "Mortgage Loan",
            "Auto Loan"
        ]

        loan_data = (
            df["Type_of_Loan"]
            .astype("string")
            .str.replace(
                r"\s*,?\s*and\s*",
                ", ",
                regex=True
            )
            .str.split(", ")
        )

        mlb = MultiLabelBinarizer(
            classes=main_loans
        )

        encoded = pd.DataFrame(
            mlb.fit_transform(loan_data),
            columns=[
                f"Has_{loan.replace('-', '').replace(' ', '_')}"
                for loan in main_loans
            ],
            index=df.index
        )

        df = pd.concat(
            [
                df.drop(columns=["Type_of_Loan"]),
                encoded
            ],
            axis=1
        )

        # =========================
        # CREDIT HISTORY
        # =========================

        df["Credit_History_Age"] = (
            df["Credit_History_Age"]
            .fillna("0 Years and 0 Months")
        )

        age_parts = (
            df["Credit_History_Age"]
            .str.extract(
                r"(?P<years>\d+)\s+Years\s+and\s+(?P<months>\d+)\s+Months"
            )
        )

        df["Credit_History_Age_Months"] = (
            pd.to_numeric(age_parts["years"])
            .fillna(0)
            * 12
            +
            pd.to_numeric(age_parts["months"])
            .fillna(0)
        )

        df = df.drop(
            columns=["Credit_History_Age"]
        )

        # =========================
        # PAYMENT BEHAVIOUR
        # =========================

        df["Spent_Frequency"] = (
            df["Payment_Behaviour"]
            .str.extract(
                r"^(Low|High)_spent"
            )
        )

        df["Payment_Value"] = (
            df["Payment_Behaviour"]
            .str.extract(
                r"(Small|Medium|Large)_value"
            )
        )

        df = df.drop(
            columns=["Payment_Behaviour"]
        )

        return df
    
    def cap_outliers(self, df):

        num_cols = [
            'Annual_Income',
            'Monthly_Inhand_Salary',
            'Num_Bank_Accounts',
            'Num_Credit_Card',
            'Interest_Rate',
            'Num_of_Loan',
            'Delay_from_due_date',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Num_Credit_Inquiries',
            'Outstanding_Debt',
            'Credit_Utilization_Ratio',
            'Total_EMI_per_month',
            'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        normal_cols = {
            "Monthly_Inhand_Salary",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio"
        }

        capping_cols = [
            col
            for col in num_cols
            if col not in normal_cols
            and col in df.columns
        ]

        for col in capping_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = max(
                0,
                q1 - 1.5 * iqr
            )

            upper = (
                q3 + 1.5 * iqr
            )

            df[col] = df[col].clip(
                lower=lower,
                upper=upper
            )

        return df

class CreditScorePreprocessor:
    """
    Handles:
    - Data cleaning
    - Data type conversion
    - Outlier capping
    - Train-test split
    - Feature engineering
    - Preprocessing pipeline creation
    - Target encoding
    """

    # Buat constructor
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 580,
        artifact_path: str | Path = "artifacts"
    ):

        self.test_size = test_size
        self.random_state = random_state

        self.artifact_dir = Path(artifact_path)

        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.main_loans = [
            "Not Specified",
            "Credit-Builder Loan",
            "Personal Loan",
            "Debt Consolidation Loan",
            "Student Loan",
            "Payday Loan",
            "Home Equity Loan",
            "Mortgage Loan",
            "Auto Loan"
        ]

        self.target_mapping = {
            "Poor": 0,
            "Standard": 1,
            "Good": 2
        }

    # ============ 1. Data Cleaning ============

    def clean_data(self, df: pd.DataFrame):

        df = df.copy()

        # Drop uninformative columns
        uninformative = [
            "Unnamed: 0",
            "ID",
            "Customer_ID",
            "Month",
            "Name",
            "Age",
            "SSN"
        ]

        df = df.drop(
            columns=uninformative,
            errors="ignore"
        )


        # Replace anomaly values pada category
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan, regex=False)
        df["Payment_of_Min_Amount"] = (df["Payment_of_Min_Amount"].replace("NM", np.nan))
        df["Payment_Behaviour"] = (df["Payment_Behaviour"].replace("!@9#%8", np.nan))
        df["Amount_invested_monthly"] = (df["Amount_invested_monthly"].replace("__10000__", np.nan))
        df["Monthly_Balance"] = (df["Monthly_Balance"].replace("__-333333333333333333333333333__",np.nan))

        # handle trailing space pada kolom numerik
        numeric_object_cols = [
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly"
        ]

        for col in numeric_object_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace("_", "", regex=False)
            )

        # hapus whitespace pada data
        df["Changed_Credit_Limit"] = (df["Changed_Credit_Limit"].replace(["", "_"], np.nan))

        # konvert datatype object pada kolom menjadi numerik
        num_convert = [
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in num_convert:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    # ============ 2. Handle Outlier ============
    def cap_column(self,df: pd.DataFrame,col: str,factor: float = 1.5,allow_negative: bool = False):

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        IQR = q3 - q1

        lower = q1 - factor * IQR
        upper = q3 + factor * IQR

        if not allow_negative:
            lower = max(0, lower)

        df[col] = df[col].clip(
            lower=lower,
            upper=upper
        )

        return df

    def cap_outliers(self, df: pd.DataFrame):

        numeric_cols = [
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        normal_cols = {
            "Monthly_Inhand_Salary",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio"
        }

        capping_cols = [
            col
            for col in numeric_cols
            if col not in normal_cols
        ]

        for col in capping_cols:

            df = self.cap_column(
                df,
                col,
                allow_negative=False
            )

        return df

    # ============ 3. Feature Engineering ============ 
    def prepare_loan_features(self, df: pd.DataFrame):

        loan_data = (
            df["Type_of_Loan"].astype("string").str.replace(
                r"\s*,?\s*and\s*",", ", regex=True
            ).str.split(", ")
        )

        mlb = MultiLabelBinarizer(classes=self.main_loans)

        encoded = pd.DataFrame(
            mlb.fit_transform(loan_data),
            columns=[
                f"Has_{loan.replace('-', '').replace(' ', '_')}"
                for loan in self.main_loans
            ],
            index=df.index
        )

        return pd.concat([df.drop(columns=["Type_of_Loan"]), encoded], axis=1)

    def convert_credit_history_to_months(self, df: pd.DataFrame):

        age_parts = (
            df["Credit_History_Age"].str.strip().str.replace(
                r"\s+"," ",regex=True
            ).str.extract(
                r"(?P<years>\d+)\s+Years\s+and\s+(?P<months>\d+)\s+Months"
            )
        )

        df["Credit_History_Age_Months"] = (
            pd.to_numeric(age_parts["years"], errors="coerce").fillna(0) * 12
            + pd.to_numeric(age_parts["months"], errors="coerce").fillna(0)
        ).astype(int)

        return df


    # ============ 4. Preprocessor ============ 
    def build_transformer(self, X_train: pd.DataFrame) -> ColumnTransformer:

        median_cols = [
            "Annual_Income",
            "Amount_invested_monthly",
            "Credit_History_Age_Months"
        ]

        mean_cols = [
            "Monthly_Inhand_Salary",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Monthly_Balance"
        ]

        loan_cols = [
            col
            for col in X_train.columns
            if col.startswith("Has_")
        ]

        ord_cols = [
            "Credit_Mix",
            "Payment_of_Min_Amount",
            "Spent_Frequency",
            "Payment_Value"
        ]

        ohe_cols = [
            "Occupation"
        ]

        median_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        mean_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler())
        ])

        loan_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        ordinal_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder",OrdinalEncoder(categories=[
                        ["Bad", "Standard", "Good"],
                        ["No", "Yes"],
                        ["Low", "High"],
                        ["Small", "Medium", "Large"]
                    ]
                )
            )
        ])

        ohe_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer(
            transformers=[
                ("median_num", median_preprocess, median_cols),
                ("mean_num", mean_preprocess, mean_cols),
                ("loan", loan_preprocess, loan_cols),
                ("ord", ordinal_preprocess, ord_cols),
                ("ohe",ohe_preprocess,ohe_cols)
            ],
            remainder="drop"
        )

    # ============ 5. Main Execution ============ 
    def run(self, data_path: str | Path):

        print("--- Step 2: Preprocessing ---")

        df = pd.read_csv(data_path)

        # Split Feature & Target
        x = df.drop(columns=["Credit_Score"])

        y = df["Credit_Score"]

        x_train, x_test, y_train, y_test = (
            train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)
        )

        # Panggil feature engineering
        feature_engineer = CreditScoreFeatureEngineer()

        sample_train = feature_engineer.fit_transform(
            x_train.copy()
        )

        # Build Transformer
        preprocess = self.build_transformer(x_train)

        # Encode Target
        y_train = y_train.map(self.target_mapping)
        y_test = y_test.map(self.target_mapping)

        return (
            x_train,
            x_test,
            y_train,
            y_test,
            preprocess
        )