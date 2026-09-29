import html
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    "<h1 style='text-align:center;'>🏠 AI House Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Machine Learning + Explainable AI + Hyperparameter Tuning"
    "</p>",
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("house_data.csv")

    data = data.dropna()

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "❌ house_data.csv not found. "
        "Keep house_data.csv in the same folder as app.py."
    )

    st.stop()


# =========================================================
# TARGET
# =========================================================

target_column = "Price"

if target_column not in df.columns:

    st.error(
        "❌ 'Price' column not found in dataset."
    )

    st.stop()


# =========================================================
# NUMERIC FEATURES
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_features = numeric_columns.copy()


feature_columns = [
    col
    for col in numeric_columns
    if col != target_column
]


if len(feature_columns) == 0:

    st.error(
        "❌ No numeric feature columns found."
    )

    st.stop()


X = df[feature_columns]

y = df[target_column]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


# =========================================================
# BASELINE MODELS
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=150,
            random_state=42
        )
}


# =========================================================
# TRAIN BASELINE MODELS
# =========================================================

results = {}

trained_models = {}

predictions = {}


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results[name] = {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    }

    trained_models[name] = model

    predictions[name] = pred


# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
).T


results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


# =========================================================
# BEST BASELINE MODEL
# =========================================================

best_model_name = results_df.index[0]

best_model = trained_models[
    best_model_name
]

best_prediction = predictions[
    best_model_name
]

best_r2 = results_df.loc[
    best_model_name,
    "R2"
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "📊 Dashboard",

        "📁 Dataset Analysis",

        "📤 Custom Dataset",
        "📊 Advanced EDA",

        "📈 Visualizations",

        "🤖 Model Comparison",

        "🎯 Model Performance",

        "⭐ Feature Importance",

        "🔍 Explainable AI",

        "⚙️ Hyperparameter Tuning",

        "💰 Price Prediction",

        "📝 Prediction History",
        "📄 Prediction Report",
        "🤖 AI House Price Assistant"

    ]

)


# =========================================================
# SESSION STATE
# =========================================================

if "last_prediction_report" not in st.session_state:
    st.session_state.last_prediction_report = None



# =========================================================
# AI HOUSE PRICE ASSISTANT
# =========================================================

def house_price_assistant(question):
    question = question.lower().strip()

    if "best model" in question or "best algorithm" in question:
        r2 = results_df.loc[best_model_name, "R2"]
        return f"🤖 Based on the current evaluation, **{best_model_name}** has the highest baseline R² score of **{r2:.4f}**."

    if "accuracy" in question or "performance" in question or "r2" in question or "score" in question:
        r2 = results_df.loc[best_model_name, "R2"]
        mae = results_df.loc[best_model_name, "MAE"]
        rmse = results_df.loc[best_model_name, "RMSE"]
        return f"📊 **{best_model_name} Performance**\n\n- R² Score: **{r2:.4f}**\n- MAE: **{mae:,.2f}**\n- RMSE: **{rmse:,.2f}**"

    if "predicted price" in question or "latest prediction" in question or question in {"prediction", "price"}:
        report = st.session_state.last_prediction_report
        if report:
            return f"🏠 Latest predicted house price: **₹{report['prediction']:,.2f}**\n\nModel: **{report['model']}**\nGenerated: **{report['timestamp']}**"
        return "💡 No prediction is available yet. Go to **💰 Price Prediction** and make a prediction first."

    if "important feature" in question or "most important" in question or "feature importance" in question or "affect price" in question:
        try:
            if hasattr(best_model, "feature_importances_"):
                imp = dict(zip(feature_columns, best_model.feature_importances_))
            elif hasattr(best_model, "coef_"):
                imp = dict(zip(feature_columns, np.abs(best_model.coef_)))
            else:
                return "ℹ️ Feature importance is not available for the current model."
            top = max(imp, key=imp.get)
            return f"⭐ The most influential feature in the current model is **{top}** with an importance value of approximately **{imp[top]:.4f}**."
        except Exception:
            return "⚠️ I could not calculate feature importance for the current model."

    if "dataset" in question or "data size" in question or "how many rows" in question:
        return f"📊 The current dataset contains **{df.shape[0]:,} rows** and **{df.shape[1]:,} columns**. The target is **{target_column}**, with **{len(feature_columns)} model features**."

    if "features" in question or "input columns" in question or "variables" in question:
        return "🔢 The model currently uses:\n\n" + "\n".join(f"- `{x}`" for x in feature_columns)

    if "explain my prediction" in question or "explain prediction" in question or "why this prediction" in question:
        report = st.session_state.last_prediction_report
        if not report:
            return "💡 First generate a prediction from **💰 Price Prediction**."
        return f"🏠 Latest predicted price: **₹{report['prediction']:,.2f}**\n\nModel: **{report['model']}**\n\nThe prediction was generated from the input values supplied on the Price Prediction page."

    if "help" in question or "what can you do" in question:
        return "🤖 Try asking:\n\n- Which model is best?\n- What is my predicted price?\n- How accurate is the model?\n- Which feature is most important?\n- Tell me about the dataset.\n- What features are used?\n- Explain my prediction."

    return "🤔 I could not match that question. Try asking about the **model, prediction, features, performance, or dataset**."

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "tuned_model" not in st.session_state:

    st.session_state.tuned_model = None


if "tuned_model_name" not in st.session_state:

    st.session_state.tuned_model_name = None


if "tuned_results" not in st.session_state:

    st.session_state.tuned_results = None


if "custom_dataset" not in st.session_state:

    st.session_state.custom_dataset = None


if "custom_target" not in st.session_state:

    st.session_state.custom_target = None


if "custom_features" not in st.session_state:

    st.session_state.custom_features = []


if "custom_models" not in st.session_state:

    st.session_state.custom_models = {}


if "custom_results" not in st.session_state:

    st.session_state.custom_results = None


if "custom_best_model" not in st.session_state:

    st.session_state.custom_best_model = None


if "custom_best_model_name" not in st.session_state:

    st.session_state.custom_best_model_name = None


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header("📊 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Houses",
            len(df)
        )

    with col2:

        st.metric(
            "Features",
            len(feature_columns)
        )

    with col3:

        st.metric(
            "Best Baseline Model",
            best_model_name
        )

    with col4:

        st.metric(
            "R² Score",
            f"{best_r2:.3f}"
        )


    st.divider()

    st.subheader("🚀 AI Features")

    features = [

        "🤖 Multiple Machine Learning Models",

        "⭐ Feature Importance",

        "🔍 Explainable AI with SHAP",

        "⚙️ Hyperparameter Tuning",

        "💰 House Price Prediction",

        "📝 Prediction History",
        "📄 Prediction Report",
        "🤖 AI House Price Assistant"

    ]

    for item in features:

        st.write(
            f"• {item}"
        )


# =========================================================
# DATASET ANALYSIS
# =========================================================

elif page == "📁 Dataset Analysis":

    st.header("📁 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )


    st.subheader(
        "📊 Statistical Summary"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# =========================================================
# VISUALIZATIONS
# =========================================================


elif page == "📊 Advanced EDA":
    st.header("📊 Advanced Exploratory Data Analysis")
    st.write("Explore distributions, outliers, correlations, and statistical relationships in the house-price dataset.")

    # Overview metrics
    st.subheader("📌 Dataset Overview")
    missing_values = int(df.isnull().sum().sum())
    numeric_count = len(numeric_features)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric Features", f"{numeric_count:,}")
    c4.metric("Missing Values", f"{missing_values:,}")

    st.divider()

    # Numeric feature selector
    if numeric_features:
        selected_feature = st.selectbox(
            "🔎 Select a numeric feature for detailed analysis",
            numeric_features,
            key="advanced_eda_feature"
        )

        feature_data = df[selected_feature].dropna()

        # Feature distribution
        st.subheader(f"📈 Distribution of {selected_feature}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(feature_data, bins=30, edgecolor="black")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {selected_feature}")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        # Outlier detection
        st.subheader(f"🚨 Outlier Detection — {selected_feature}")
        q1 = feature_data.quantile(0.25)
        q3 = feature_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = feature_data[
            (feature_data < lower_bound) | (feature_data > upper_bound)
        ]

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Q1", f"{q1:,.2f}")
        o2.metric("Q3", f"{q3:,.2f}")
        o3.metric("IQR", f"{iqr:,.2f}")
        o4.metric("Outliers", f"{len(outliers):,}")

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.boxplot(feature_data, vert=False)
        ax.set_xlabel(selected_feature)
        ax.set_title(f"Box Plot — {selected_feature}")
        st.pyplot(fig)
        plt.close(fig)

        if len(outliers) > 0:
            st.info(
                f"{len(outliers):,} potential outlier(s) detected using the IQR method."
            )
        else:
            st.success("No potential outliers detected using the IQR method.")

        st.divider()

        # Correlation ranking with target
        st.subheader("🎯 Price Correlation Ranking")

        correlation_data = df[numeric_features].corr()[target_column].drop(
            labels=[target_column], errors="ignore"
        ).sort_values(ascending=False)

        corr_table = pd.DataFrame({
            "Feature": correlation_data.index,
            "Correlation with Price": correlation_data.values,
            "Absolute Correlation": correlation_data.abs().values
        })

        corr_table["Correlation with Price"] = corr_table[
            "Correlation with Price"
        ].round(4)
        corr_table["Absolute Correlation"] = corr_table[
            "Absolute Correlation"
        ].round(4)

        st.dataframe(
            corr_table,
            use_container_width=True,
            hide_index=True
        )

        if not correlation_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_corr = correlation_data.sort_values()
            ax.barh(sorted_corr.index, sorted_corr.values)
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("Correlation Coefficient")
            ax.set_title("Feature Correlation with House Price")
            ax.grid(axis="x", alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # Feature vs Price analysis
        st.subheader(f"🏠 {selected_feature} vs {target_column}")

        plot_df = df[[selected_feature, target_column]].dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(
            plot_df[selected_feature],
            plot_df[target_column],
            alpha=0.65
        )

        # Add a simple linear trend line when enough data exists
        if len(plot_df) >= 2 and plot_df[selected_feature].nunique() > 1:
            x = plot_df[selected_feature].to_numpy()
            y = plot_df[target_column].to_numpy()

            coefficients = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            trend_y = np.polyval(coefficients, trend_x)

            ax.plot(trend_x, trend_y, linewidth=2)

        ax.set_xlabel(selected_feature)
        ax.set_ylabel(target_column)
        ax.set_title(f"{selected_feature} vs {target_column}")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        # Statistics
        st.subheader(f"📊 Statistical Summary — {selected_feature}")

        stats = pd.DataFrame({
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Standard Deviation",
                "Minimum",
                "Q1 (25%)",
                "Q3 (75%)",
                "Maximum"
            ],
            "Value": [
                feature_data.count(),
                feature_data.mean(),
                feature_data.median(),
                feature_data.std(),
                feature_data.min(),
                q1,
                q3,
                feature_data.max()
            ]
        })

        st.dataframe(
            stats.style.format({"Value": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        # Full numeric statistical summary
        st.subheader("📋 Complete Numeric Statistical Summary")
        summary = df[numeric_features].describe().T
        summary = summary.rename(columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Minimum",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Maximum"
        })

        st.dataframe(
            summary.style.format("{:,.2f}"),
            use_container_width=True
        )

elif page == "📤 Custom Dataset":

    st.header("📤 Custom Dataset Upload")
    st.write("Upload your own CSV dataset and train regression models on it.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="custom_csv_uploader"
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success(f"Dataset uploaded successfully: {uploaded_file.name}")

            st.subheader("👀 Dataset Preview")
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{uploaded_df.shape[0]:,}")
            m2.metric("Columns", f"{uploaded_df.shape[1]:,}")
            m3.metric("Missing Values", f"{int(uploaded_df.isnull().sum().sum()):,}")

            st.divider()

            columns = uploaded_df.columns.tolist()
            target = st.selectbox(
                "🎯 Select Target Column",
                columns,
                key="custom_target_select"
            )

            numeric_cols = uploaded_df.select_dtypes(include=np.number).columns.tolist()
            feature_options = [c for c in numeric_cols if c != target]

            if target not in numeric_cols:
                st.warning(
                    "The selected target column is not numeric. "
                    "For this regression application, please select a numeric target column."
                )
                st.stop()

            if not feature_options:
                st.error("❌ No numeric feature columns are available after excluding the target.")
                st.stop()

            selected_features = st.multiselect(
                "🔢 Select Numeric Feature Columns",
                feature_options,
                default=feature_options,
                key="custom_feature_select"
            )

            if not selected_features:
                st.warning("Select at least one feature column to continue.")
                st.stop()

            clean_df = uploaded_df[selected_features + [target]].copy()
            before_rows = len(clean_df)
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna()
            removed_rows = before_rows - len(clean_df)

            if removed_rows > 0:
                st.info(f"🧹 Basic cleaning removed {removed_rows:,} row(s) containing missing/infinite values.")

            if len(clean_df) < 10:
                st.error("❌ At least 10 clean rows are recommended for model training.")
                st.stop()

            st.subheader("📊 Clean Dataset")
            st.dataframe(clean_df.head(10), use_container_width=True)

            train_clicked = st.button(
                "🤖 Train Models on Custom Dataset",
                type="primary",
                use_container_width=True
            )

            if train_clicked:
                custom_X = clean_df[selected_features]
                custom_y = clean_df[target]

                custom_X_train, custom_X_test, custom_y_train, custom_y_test = train_test_split(
                    custom_X,
                    custom_y,
                    test_size=0.20,
                    random_state=42
                )

                custom_model_definitions = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=150,
                        random_state=42
                    ),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "Extra Trees": ExtraTreesRegressor(
                        n_estimators=150,
                        random_state=42
                    )
                }

                custom_results = {}
                custom_trained_models = {}

                with st.spinner("Training models on your custom dataset..."):
                    for model_name, model in custom_model_definitions.items():
                        model.fit(custom_X_train, custom_y_train)
                        predictions = model.predict(custom_X_test)

                        mae = mean_absolute_error(custom_y_test, predictions)
                        rmse = np.sqrt(mean_squared_error(custom_y_test, predictions))
                        r2 = r2_score(custom_y_test, predictions)

                        custom_results[model_name] = {
                            "MAE": mae,
                            "RMSE": rmse,
                            "R² Score": r2
                        }
                        custom_trained_models[model_name] = model

                best_custom_name = max(
                    custom_results,
                    key=lambda name: custom_results[name]["R² Score"]
                )

                st.session_state.custom_dataset = clean_df
                st.session_state.custom_target = target
                st.session_state.custom_features = selected_features
                st.session_state.custom_models = custom_trained_models
                st.session_state.custom_results = custom_results
                st.session_state.custom_best_model = custom_trained_models[best_custom_name]
                st.session_state.custom_best_model_name = best_custom_name

                st.success(f"Training completed. Best model by R²: {best_custom_name}")

            if st.session_state.custom_results is not None:
                # Only show results for the currently selected uploaded dataset.
                if (
                    st.session_state.custom_target == target
                    and st.session_state.custom_dataset is not None
                    and st.session_state.custom_features == selected_features
                ):
                    st.divider()
                    st.subheader("📈 Custom Dataset Model Comparison")

                    custom_results_df = pd.DataFrame(
                        st.session_state.custom_results
                    ).T.reset_index().rename(columns={"index": "Model"})

                    st.dataframe(
                        custom_results_df.style.format({
                            "MAE": "{:,.4f}",
                            "RMSE": "{:,.4f}",
                            "R² Score": "{:,.4f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("🏆 Best Custom Dataset Model")
                    st.info(
                        f"{st.session_state.custom_best_model_name} "
                        f"selected using the highest R² score."
                    )

                    st.subheader("💰 Custom Dataset Prediction")
                    st.write("Enter values for the selected features:")

                    input_values = {}
                    input_cols = st.columns(2)

                    for idx, feature in enumerate(selected_features):
                        feature_series = clean_df[feature]
                        default_value = float(feature_series.median())
                        min_value = float(feature_series.min())
                        max_value = float(feature_series.max())

                        with input_cols[idx % 2]:
                            input_values[feature] = st.number_input(
                                feature,
                                min_value=min_value,
                                max_value=max_value,
                                value=default_value,
                                key=f"custom_prediction_{feature}"
                            )

                    if st.button(
                        "🔮 Predict Custom Dataset Value",
                        use_container_width=True
                    ):
                        input_df = pd.DataFrame([input_values])
                        custom_prediction = st.session_state.custom_best_model.predict(input_df)[0]

                        st.success(
                            f"Predicted {target}: {custom_prediction:,.2f}"
                        )

        except Exception as e:
            st.error(f"❌ Could not process the uploaded dataset: {e}")

elif page == "📈 Visualizations":

    st.header(
        "📈 Data Visualizations"
    )


    if "Area" in df.columns:

        st.subheader(
            "🏠 Area vs Price"
        )

        fig, ax = plt.subplots()

        ax.scatter(
            df["Area"],
            df["Price"]
        )

        ax.set_xlabel(
            "Area"
        )

        ax.set_ylabel(
            "Price"
        )

        ax.set_title(
            "Area vs House Price"
        )

        st.pyplot(fig)

        plt.close(fig)


    st.subheader(
        "🔗 Correlation Matrix"
    )

    correlation = df[
        numeric_columns
    ].corr()

    fig, ax = plt.subplots()

    im = ax.imshow(
        correlation,
        cmap="coolwarm"
    )

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    fig.colorbar(im)

    st.pyplot(fig)

    plt.close(fig)


    st.subheader(
        "🎯 Actual vs Predicted"
    )

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        best_prediction
    )

    ax.set_xlabel(
        "Actual Price"
    )

    ax.set_ylabel(
        "Predicted Price"
    )

    ax.set_title(
        f"{best_model_name}: Actual vs Predicted"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.header(
        "🤖 Machine Learning Model Comparison"
    )

    display_results = results_df.copy()

    display_results["MAE"] = display_results[
        "MAE"
    ].round(2)

    display_results["RMSE"] = display_results[
        "RMSE"
    ].round(2)

    display_results["R2"] = display_results[
        "R2"
    ].round(4)


    st.dataframe(
        display_results,
        use_container_width=True
    )


    st.subheader(
        "📊 R² Comparison"
    )

    fig, ax = plt.subplots()

    ax.bar(
        results_df.index,
        results_df["R2"]
    )

    ax.set_ylabel(
        "R² Score"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_title(
        "Model R² Comparison"
    )

    plt.xticks(
        rotation=30
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🎯 Model Performance":

    st.header(
        "🎯 Model Performance"
    )

    st.success(
        f"Best Baseline Model: {best_model_name}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "MAE",
            f"{results_df.loc[best_model_name, 'MAE']:.2f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{results_df.loc[best_model_name, 'RMSE']:.2f}"
        )


    with col3:

        st.metric(
            "R²",
            f"{best_r2:.4f}"
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "⭐ Feature Importance":

    st.header(
        "⭐ Feature Importance"
    )

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({

            "Feature":
                feature_columns,

            "Importance":
                best_model.feature_importances_

        }).sort_values(
            by="Importance",
            ascending=False
        )


        st.dataframe(
            importance_df,
            use_container_width=True
        )


        fig, ax = plt.subplots()

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Feature Importance"
        )

        ax.invert_yaxis()

        st.pyplot(fig)

        plt.close(fig)


    elif hasattr(
        best_model,
        "coef_"
    ):

        coefficient_df = pd.DataFrame({

            "Feature":
                feature_columns,

            "Coefficient":
                best_model.coef_

        })

        coefficient_df[
            "Absolute Impact"
        ] = np.abs(
            coefficient_df["Coefficient"]
        )

        coefficient_df = coefficient_df.sort_values(
            by="Absolute Impact",
            ascending=False
        )

        st.dataframe(
            coefficient_df,
            use_container_width=True
        )


    else:

        st.warning(
            "Feature importance unavailable."
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🔍 Explainable AI":

    st.header(
        "🔍 Explainable AI — SHAP"
    )


    if not SHAP_AVAILABLE:

        st.error(
            "SHAP is not installed."
        )

        st.code(
            "python -m pip install shap"
        )

        st.stop()


    try:

        if best_model_name in [

            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "Extra Trees"

        ]:

            explainer = shap.TreeExplainer(
                best_model
            )

        else:

            explainer = shap.LinearExplainer(
                best_model,
                X_train
            )


        shap_values = explainer(
            X_test
        )


        st.subheader(
            "📊 Global Feature Impact"
        )

        shap.plots.bar(
            shap_values,
            show=False
        )

        st.pyplot(
            plt.gcf()
        )

        plt.close()


        mean_abs_shap = np.abs(
            shap_values.values
        ).mean(axis=0)


        shap_df = pd.DataFrame({

            "Feature":
                feature_columns,

            "Mean Absolute SHAP":
                mean_abs_shap

        }).sort_values(
            by="Mean Absolute SHAP",
            ascending=False
        )


        st.subheader(
            "Feature Impact Table"
        )

        st.dataframe(
            shap_df,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"SHAP error: {e}"
        )


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

elif page == "⚙️ Hyperparameter Tuning":

    st.header(
        "⚙️ Hyperparameter Tuning"
    )

    st.write(
        "GridSearchCV will test multiple parameter "
        "combinations and select the configuration "
        "with the highest cross-validation R² score."
    )


    tuning_model = st.selectbox(

        "Select Model to Tune",

        [

            "Random Forest",
            "Gradient Boosting",
            "Extra Trees",
            "Decision Tree"

        ]

    )


    if tuning_model == "Random Forest":

        base_model = RandomForestRegressor(
            random_state=42
        )

        param_grid = {

            "n_estimators":
                [100, 200],

            "max_depth":
                [None, 10, 20],

            "min_samples_split":
                [2, 5]

        }


    elif tuning_model == "Gradient Boosting":

        base_model = GradientBoostingRegressor(
            random_state=42
        )

        param_grid = {

            "n_estimators":
                [100, 200],

            "learning_rate":
                [0.05, 0.1],

            "max_depth":
                [2, 3]

        }


    elif tuning_model == "Extra Trees":

        base_model = ExtraTreesRegressor(
            random_state=42
        )

        param_grid = {

            "n_estimators":
                [100, 200],

            "max_depth":
                [None, 10, 20],

            "min_samples_split":
                [2, 5]

        }


    else:

        base_model = DecisionTreeRegressor(
            random_state=42
        )

        param_grid = {

            "max_depth":
                [None, 5, 10, 20],

            "min_samples_split":
                [2, 5, 10],

            "min_samples_leaf":
                [1, 2, 4]

        }


    st.subheader(
        "🔧 Parameter Search Space"
    )

    st.json(
        param_grid
    )


    if st.button(
        "🚀 Start Hyperparameter Tuning",
        use_container_width=True
    ):

        with st.spinner(
            "Training and testing parameter combinations..."
        ):

            grid_search = GridSearchCV(

                estimator=base_model,

                param_grid=param_grid,

                cv=5,

                scoring="r2",

                n_jobs=-1

            )

            grid_search.fit(
                X_train,
                y_train
            )


        tuned_model = grid_search.best_estimator_

        tuned_prediction = tuned_model.predict(
            X_test
        )


        tuned_mae = mean_absolute_error(
            y_test,
            tuned_prediction
        )


        tuned_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                tuned_prediction
            )
        )


        tuned_r2 = r2_score(
            y_test,
            tuned_prediction
        )


        st.session_state.tuned_model = tuned_model

        st.session_state.tuned_model_name = tuning_model

        st.session_state.tuned_results = {

            "MAE": tuned_mae,

            "RMSE": tuned_rmse,

            "R2": tuned_r2,

            "CV R2": grid_search.best_score_,

            "Best Parameters":
                grid_search.best_params_

        }


        st.success(
            "✅ Hyperparameter tuning completed!"
        )


    # -----------------------------------------------------
    # SHOW TUNING RESULTS
    # -----------------------------------------------------

    if st.session_state.tuned_results is not None:

        tuning_results = (
            st.session_state.tuned_results
        )


        st.subheader(
            "🏆 Tuned Model Results"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Test MAE",
                f"{tuning_results['MAE']:.2f}"
            )


        with col2:

            st.metric(
                "Test RMSE",
                f"{tuning_results['RMSE']:.2f}"
            )


        with col3:

            st.metric(
                "Test R²",
                f"{tuning_results['R2']:.4f}"
            )


        with col4:

            st.metric(
                "CV R²",
                f"{tuning_results['CV R2']:.4f}"
            )


        st.subheader(
            "🔧 Best Parameters"
        )


        for parameter, value in tuning_results[
            "Best Parameters"
        ].items():

            st.write(
                f"**{parameter}:** {value}"
            )


        # -------------------------------------------------
        # BASELINE VS TUNED
        # -------------------------------------------------

        st.subheader(
            "📊 Baseline vs Tuned Model"
        )


        baseline_r2 = results_df.loc[
            tuning_model,
            "R2"
        ]


        comparison = pd.DataFrame({

            "Model": [

                f"{tuning_model} - Baseline",

                f"{tuning_model} - Tuned"

            ],

            "R² Score": [

                baseline_r2,

                tuning_results["R2"]

            ]

        })


        st.dataframe(
            comparison,
            use_container_width=True
        )


        fig, ax = plt.subplots()

        ax.bar(
            comparison["Model"],
            comparison["R² Score"]
        )

        ax.set_ylabel(
            "R² Score"
        )

        ax.set_title(
            "Baseline vs Tuned Model"
        )

        plt.xticks(
            rotation=15
        )

        st.pyplot(fig)

        plt.close(fig)


# =========================================================
# PRICE PREDICTION
# =========================================================

elif page == "💰 Price Prediction":

    st.header(
        "💰 House Price Prediction"
    )


    prediction_model = best_model


    if st.session_state.tuned_model is not None:

        use_tuned = st.checkbox(
            "Use tuned model for prediction",
            value=True
        )

        if use_tuned:

            prediction_model = (
                st.session_state.tuned_model
            )

            st.info(
                f"Using tuned "
                f"{st.session_state.tuned_model_name} model."
            )


    input_values = {}


    for feature in feature_columns:

        input_values[feature] = st.number_input(

            feature,

            min_value=0.0,

            value=float(
                df[feature].median()
            )

        )


    if st.button(
        "🔮 Predict House Price",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [input_values]
        )


        prediction = prediction_model.predict(
            input_df
        )[0]


        st.success(
            f"🏠 Estimated House Price: "
            f"₹{prediction:,.2f}"
        )


        history = input_values.copy()

        history["Predicted Price"] = prediction

        history["Model"] = (
            st.session_state.tuned_model_name
            if (
                st.session_state.tuned_model is not None
                and prediction_model is st.session_state.tuned_model
            )
            else best_model_name
        )


        st.session_state.prediction_history.append(
            history
        )

        # Save the latest prediction for the automatic report generator.
        model_used = (
            st.session_state.tuned_model_name
            if (
                st.session_state.tuned_model is not None
                and prediction_model is st.session_state.tuned_model
            )
            else best_model_name
        )

        st.session_state.last_prediction_report = {
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "model": model_used,
            "prediction": float(prediction),
            "inputs": input_values.copy(),
            "r2": float(
                results_df.loc[best_model_name, "R2"]
            ),
            "mae": float(
                results_df.loc[best_model_name, "MAE"]
            ),
            "rmse": float(
                results_df.loc[best_model_name, "RMSE"]
            ),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "features": int(len(feature_columns))
        }

        st.info(
            "📄 Prediction report is ready. "
            "Open the 'Prediction Report' page from the sidebar."
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================


# =========================================================
# AUTOMATIC PREDICTION REPORT
# =========================================================

elif page == "📄 Prediction Report":

    st.header("📄 Automatic Prediction Report")

    report = st.session_state.last_prediction_report

    if report is None:

        st.warning(
            "No prediction report is available yet."
        )

        st.info(
            "Go to '💰 Price Prediction', make a prediction, "
            "and then return here to generate the report."
        )

    else:

        st.success(
            "✅ Latest prediction report is ready."
        )

        st.subheader("🏠 Prediction Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Estimated House Price",
            f"₹{report['prediction']:,.2f}"
        )

        c2.metric(
            "Model",
            report["model"]
        )

        c3.metric(
            "Generated At",
            report["timestamp"]
        )

        st.divider()

        st.subheader("🤖 Model Performance")

        performance_df = pd.DataFrame({
            "Metric": [
                "R² Score",
                "MAE",
                "RMSE"
            ],
            "Value": [
                report["r2"],
                report["mae"],
                report["rmse"]
            ]
        })

        st.dataframe(
            performance_df.style.format({
                "Value": "{:,.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🏡 Input Features")

        input_report_df = pd.DataFrame({
            "Feature": list(report["inputs"].keys()),
            "Input Value": list(report["inputs"].values())
        })

        st.dataframe(
            input_report_df.style.format({
                "Input Value": "{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📊 Dataset Information")

        d1, d2, d3 = st.columns(3)

        d1.metric("Dataset Rows", f"{report['rows']:,}")
        d2.metric("Dataset Columns", f"{report['columns']:,}")
        d3.metric("Model Features", f"{report['features']:,}")

        # Build a downloadable HTML report.
        input_rows = "".join(
            f"""
            <tr>
                <td>{html.escape(str(feature))}</td>
                <td>{value:,.2f}</td>
            </tr>
            """
            for feature, value in report["inputs"].items()
        )

        html_report = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI House Price Prediction Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.5;
}}
h1, h2 {{
    margin-bottom: 8px;
}}
.card {{
    padding: 18px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin-bottom: 20px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
}}
th {{
    background: #f2f2f2;
}}
.price {{
    font-size: 28px;
    font-weight: bold;
}}
</style>
</head>

<body>

<h1>🏠 AI House Price Prediction Report</h1>

<div class="card">
<h2>Prediction Summary</h2>
<p><strong>Generated At:</strong> {html.escape(report["timestamp"])}</p>
<p><strong>Model:</strong> {html.escape(report["model"])}</p>
<p class="price">
Estimated House Price: ₹{report["prediction"]:,.2f}
</p>
</div>

<div class="card">
<h2>Model Performance</h2>
<table>
<tr>
<th>Metric</th>
<th>Value</th>
</tr>
<tr>
<td>R² Score</td>
<td>{report["r2"]:.4f}</td>
</tr>
<tr>
<td>MAE</td>
<td>{report["mae"]:,.4f}</td>
</tr>
<tr>
<td>RMSE</td>
<td>{report["rmse"]:,.4f}</td>
</tr>
</table>
</div>

<div class="card">
<h2>Input Features</h2>
<table>
<tr>
<th>Feature</th>
<th>Input Value</th>
</tr>
{input_rows}
</table>
</div>

<div class="card">
<h2>Dataset Information</h2>
<p><strong>Rows:</strong> {report["rows"]:,}</p>
<p><strong>Columns:</strong> {report["columns"]:,}</p>
<p><strong>Model Features:</strong> {report["features"]:,}</p>
</div>

<p>
Generated by AI House Price Prediction application.
</p>

</body>
</html>
"""

        st.download_button(
            label="📥 Download Prediction Report",
            data=html_report,
            file_name="house_price_prediction_report.html",
            mime="text/html",
            use_container_width=True
        )

        st.caption(
            "The downloaded HTML report can be opened in any web browser "
            "and printed/saved as PDF if needed."
        )

elif page == "🤖 AI House Price Assistant":
    st.header("🤖 AI House Price Assistant")
    st.write("Ask questions about your house-price model, dataset, features, performance, or latest prediction.")
    st.divider()

    st.subheader("💡 Example Questions")
    examples = [
        "Which model is best?",
        "What is my predicted price?",
        "How accurate is the model?",
        "Which feature is most important?",
        "Tell me about the dataset.",
        "What features are used?",
        "Explain my prediction."
    ]

    if "assistant_question" not in st.session_state:
        st.session_state.assistant_question = ""

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, use_container_width=True, key=f"assistant_example_{i}"):
                st.session_state.assistant_question = example

    question = st.text_input(
        "💬 Ask your question",
        value=st.session_state.assistant_question,
        placeholder="Example: Which model is best?"
    )

    if st.button("🤖 Ask Assistant", use_container_width=True):
        if question.strip():
            st.session_state.assistant_question = question
            answer = house_price_assistant(question)
            st.subheader("🤖 Assistant Response")
            st.success(answer)
        else:
            st.warning("Please enter a question first.")

elif page == "📝 Prediction History":

    st.header("📝 Prediction History")


    if len(
        st.session_state.prediction_history
    ) == 0:

        st.info(
            "No predictions available yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )


        st.dataframe(
            history_df,
            use_container_width=True
        )


        csv = history_df.to_csv(
            index=False
        )


        st.download_button(

            "⬇️ Download CSV",

            csv,

            "prediction_history.csv",

            "text/csv"

        )


        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    "<center>"
    "🏠 AI House Price Prediction | "
    "ML + XAI + Hyperparameter Tuning"
    "</center>",
    unsafe_allow_html=True
)