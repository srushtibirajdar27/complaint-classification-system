import re
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Small sample dataset
# Replace or expand these examples with real, anonymized data.
# Each complaint is paired with a category label.
# ---------------------------------------------------------
DATA = [
    # Billing / Payment
    ("I was charged twice for the same order", "Billing / Payment"),
    ("My payment was deducted but the order failed", "Billing / Payment"),
    ("There is an incorrect amount on my bill", "Billing / Payment"),
    ("I need a refund for a duplicate transaction", "Billing / Payment"),
    ("The invoice includes extra charges", "Billing / Payment"),
    ("Money was taken from my account without confirmation", "Billing / Payment"),
    ("My refund has not arrived in my bank account", "Billing / Payment"),
    ("The subscription fee is higher than expected", "Billing / Payment"),
    ("I cannot complete payment using my card", "Billing / Payment"),
    ("Please correct the payment receipt", "Billing / Payment"),

    # Delivery / Shipping
    ("My parcel has not arrived yet", "Delivery / Shipping"),
    ("The delivery is delayed by several days", "Delivery / Shipping"),
    ("The tracking page has not updated", "Delivery / Shipping"),
    ("The courier delivered my package to the wrong address", "Delivery / Shipping"),
    ("I received an empty delivery box", "Delivery / Shipping"),
    ("Please tell me when my order will be delivered", "Delivery / Shipping"),
    ("The shipment is missing", "Delivery / Shipping"),
    ("The delivery person did not call me", "Delivery / Shipping"),
    ("My order is stuck in transit", "Delivery / Shipping"),
    ("I want to report a damaged parcel during delivery", "Delivery / Shipping"),

    # Technical Issue
    ("The application crashes whenever I open it", "Technical Issue"),
    ("I cannot log in to my account", "Technical Issue"),
    ("The website keeps showing an error message", "Technical Issue"),
    ("The app is very slow and freezes", "Technical Issue"),
    ("I cannot reset my password", "Technical Issue"),
    ("The page does not load on my laptop", "Technical Issue"),
    ("The software stopped working after the update", "Technical Issue"),
    ("I am unable to upload a document", "Technical Issue"),
    ("The mobile application closes unexpectedly", "Technical Issue"),
    ("The website button is not responding", "Technical Issue"),

    # Product Quality
    ("The product arrived broken", "Product Quality"),
    ("The item I received is defective", "Product Quality"),
    ("The product stopped working after one day", "Product Quality"),
    ("The material quality is very poor", "Product Quality"),
    ("The item is scratched and damaged", "Product Quality"),
    ("The device overheats while using it", "Product Quality"),
    ("The product does not work as advertised", "Product Quality"),
    ("The package contained a faulty item", "Product Quality"),
    ("The product has a missing part", "Product Quality"),
    ("I received a low quality product", "Product Quality"),

    # Account / Security
    ("Someone changed my account password", "Account / Security"),
    ("I think my account has been hacked", "Account / Security"),
    ("I cannot access my locked account", "Account / Security"),
    ("I received a login alert that was not me", "Account / Security"),
    ("My account details were changed without permission", "Account / Security"),
    ("I want to report suspicious activity on my profile", "Account / Security"),
    ("I cannot verify my account using the code", "Account / Security"),
    ("My account has been suspended by mistake", "Account / Security"),
    ("I received a suspicious password reset email", "Account / Security"),
    ("Please help me secure my account", "Account / Security"),

    # Customer Service
    ("The support team has not replied to my messages", "Customer Service"),
    ("The customer service agent was rude", "Customer Service"),
    ("I have been waiting on the helpline for an hour", "Customer Service"),
    ("Nobody has followed up on my complaint", "Customer Service"),
    ("The representative did not solve my problem", "Customer Service"),
    ("I was transferred between departments repeatedly", "Customer Service"),
    ("Your staff gave me incorrect information", "Customer Service"),
    ("I cannot reach customer support", "Customer Service"),
    ("The service team closed my ticket without helping", "Customer Service"),
    ("I am unhappy with the support I received", "Customer Service"),

    # Cancellation / Return
    ("I want to cancel my order", "Cancellation / Return"),
    ("Please arrange a return for this item", "Cancellation / Return"),
    ("My cancellation request has not been processed", "Cancellation / Return"),
    ("I need to exchange the product for another size", "Cancellation / Return"),
    ("The return pickup has not happened", "Cancellation / Return"),
    ("I want to return an item I purchased yesterday", "Cancellation / Return"),
    ("The seller rejected my return request", "Cancellation / Return"),
    ("I received the wrong size and need an exchange", "Cancellation / Return"),
    ("Please cancel my subscription", "Cancellation / Return"),
    ("I cannot find the option to return my order", "Cancellation / Return"),

    # Other
    ("I have a general question about your services", "Other"),
    ("Please update the contact information on your website", "Other"),
    ("I would like information about your business hours", "Other"),
    ("I want to share feedback about your company", "Other"),
    ("Where can I find your office address", "Other"),
    ("Please explain the features of this service", "Other"),
    ("I have a suggestion for improving your website", "Other"),
    ("Can you provide more information about your plans", "Other"),
    ("I need help understanding the terms and conditions", "Other"),
    ("I have a question that does not fit another category", "Other"),
]

DEPARTMENTS = {
    "Billing / Payment": "Finance / Billing Team",
    "Delivery / Shipping": "Logistics / Delivery Team",
    "Technical Issue": "Technical Support Team",
    "Product Quality": "Product / Quality Team",
    "Account / Security": "Account Security Team",
    "Customer Service": "Customer Support Team",
    "Cancellation / Return": "Returns / Order Management Team",
    "Other": "General Support Team",
}

def clean_text(text):
    """Basic text cleanup: lowercase and remove punctuation/symbols."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def train_model():
    df = pd.DataFrame(DATA, columns=["complaint", "category"])
    df["cleaned_complaint"] = df["complaint"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_complaint"],
        df["category"],
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    matrix_labels = sorted(df["category"].unique())
    matrix = confusion_matrix(y_test, predictions, labels=matrix_labels)

    return model, accuracy, report, matrix, matrix_labels, df, len(X_train), len(X_test)

st.set_page_config(page_title="Complaint Classification System", page_icon="📝", layout="wide")

st.title("📝 NLP Complaint Classification System")
st.write(
    "Enter a complaint below. The machine-learning model predicts its category "
    "and suggests the team that could handle it."
)
st.info(
    "Academic prototype: this app uses a small, hand-written sample dataset. "
    "Predictions and evaluation scores are for demonstration only, not real-world deployment."
)

model, accuracy, report, matrix, matrix_labels, df, train_count, test_count = train_model()

with st.sidebar:
    st.header("Project information")
    st.write("**NLP method:** TF-IDF")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Categories:** 8")
    st.write(f"**Sample records:** {len(df)}")
    st.caption("Use the Evaluation tab to view sample test-set metrics.")

tab_predict, tab_examples, tab_evaluation, tab_dataset = st.tabs(
    ["Classify Complaint", "Try Examples", "Model Evaluation", "Sample Dataset"]
)

with tab_predict:
    st.subheader("Enter a complaint")
    complaint = st.text_area(
        "Complaint text",
        placeholder="Example: My payment was deducted twice for the same order.",
        height=130,
        max_chars=2000,
    )

    if st.button("Classify Complaint", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint before clicking classify.")
        else:
            cleaned = clean_text(complaint)
            predicted_category = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            class_names = model.named_steps["classifier"].classes_
            confidence = float(max(probabilities)) * 100

            st.markdown("### Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted category", predicted_category)
            with col2:
                st.metric("Suggested department", DEPARTMENTS[predicted_category])

            st.caption(
                f"Model score for this prediction: {confidence:.1f}% "
                "(not a guarantee that the prediction is correct)."
            )
            st.write("**Cleaned text:**", cleaned if cleaned else "(empty)")
            st.write("**Scores by category**")
            score_df = pd.DataFrame({
                "Category": class_names,
                "Model score (%)": (probabilities * 100).round(1),
            }).sort_values("Model score (%)", ascending=False)
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            st.caption("These scores are model estimates, not calibrated probabilities.")

with tab_examples:
    st.subheader("Try a sample complaint")
    examples = {
        "Payment deducted twice": "I was charged twice for the same order.",
        "Late delivery": "My parcel has not arrived and the tracking is not updated.",
        "App problem": "The application crashes whenever I open it.",
        "Damaged product": "The product arrived broken and has a missing part.",
        "Suspicious login": "I received a login alert that was not me.",
        "Unhelpful support": "The support team has not replied to my messages.",
        "Return request": "I want to return this item and get an exchange.",
    }
    selected_example = st.selectbox("Choose an example", list(examples.keys()))
    st.text_area("Example text", examples[selected_example], height=90, disabled=True)
    if st.button("Classify Example"):
        text = examples[selected_example]
        category = model.predict([clean_text(text)])[0]
        st.success(f"Predicted category: {category}")
        st.write(f"Suggested department: **{DEPARTMENTS[category]}**")

with tab_evaluation:
    st.subheader("Model evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{accuracy * 100:.1f}%")
    c2.metric("Training records", train_count)
    c3.metric("Test records", test_count)

    st.warning(
        "The test set is very small and the examples were manually written. "
        "This accuracy should not be treated as evidence of real-world performance."
    )

    st.write("**Per-category evaluation**")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(2), use_container_width=True)

    st.write("**Confusion matrix**")
    matrix_df = pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels)
    st.dataframe(matrix_df, use_container_width=True)

with tab_dataset:
    st.subheader("Sample training dataset")
    st.write("The classifier learns from the complaint text and its labelled category.")
    st.dataframe(df[["complaint", "category"]], use_container_width=True, hide_index=True)
    csv = df[["complaint", "category"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sample dataset as CSV",
        data=csv,
        file_name="complaint_dataset.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Built with Python, Streamlit, Pandas and Scikit-learn | NLP Academic Mini-Project")
