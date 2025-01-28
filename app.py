import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# User-defined plotting function
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate the model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # Plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Streamlit App Interface
st.title("SVM Classifier with User Data")
st.write("This app allows you to explore an SVM classifier trained on user data.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(data)

    # Encode categorical data
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])  # Convert Male/Female to 0/1

    # Split features and target
    X = data[['Gender', 'Age', 'EstimatedSalary']]
    y = data['Purchased']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Model selection
    kernel_option = st.selectbox("Choose kernel type for SVC:", ["linear", "poly", "rbf"])
    if kernel_option == "poly":
        degree = st.slider("Select polynomial degree:", 2, 5, value=2)
        model = SVC(kernel=kernel_option, degree=degree)
    elif kernel_option == "rbf":
        gamma = st.slider("Select RBF gamma value:", 0.1, 50.0, value=1.0)
        model = SVC(kernel=kernel_option, gamma=gamma)
    else:
        model = SVC(kernel=kernel_option)

    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display metrics
    st.write("### Model Performance:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    # Display confusion matrix
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)

    # Optional decision boundary plot
    if st.checkbox("Show Decision Boundary (2D for linear kernel)"):
        if kernel_option == "linear" and X.shape[1] == 2:
            fig, ax = plt.subplots()
            plot_svc_decision_function(model, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Decision boundary visualization is only available for 2D linear kernels.")
