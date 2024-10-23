<h1>Heart Disease Prediction Using Machine Learning</h1>

<p>This project focuses on predicting the likelihood of heart disease based on various medical and lifestyle factors. Leveraging a range of machine learning algorithms, the goal is to build a model that can assist healthcare professionals in identifying individuals at higher risk of heart disease, allowing for timely interventions and treatment plans.</p>

<h2>Features:</h2>
<ul>
    <li><strong>Dataset:</strong> The dataset used contains medical and lifestyle variables such as age, cholesterol levels, blood pressure, glucose levels, and smoking habits. Each record corresponds to an individual's health profile and whether they have been diagnosed with heart disease.</li>
    <li><strong>Selected Features for Prediction:</strong>
        <ul>
            <li>Year (<code>yr</code>)</li>
            <li>Cholesterol level (<code>cholesterol</code>)</li>
            <li>Weight (<code>weight</code>)</li>
            <li>Glucose level (<code>gluc</code>)</li>
            <li>Diastolic blood pressure (<code>ap_lo</code>)</li>
            <li>Systolic blood pressure (<code>ap_hi</code>)</li>
            <li>Activity level (<code>active</code>)</li>
            <li>Smoking habit (<code>smoke</code>)</li>
        </ul>
    </li>
    <li><strong>Machine Learning Algorithms:</strong>
        <ul>
            <li>Logistic Regression</li>
            <li>Random Forest Classifier</li>
            <li>Support Vector Machine (SVM)</li>
            <li>K-Nearest Neighbors (KNN)</li>
            <li>Decision Trees</li>
        </ul>
    </li>
    <li><strong>Evaluation Metrics:</strong>
        <ul>
            <li>Accuracy</li>
            <li>Precision</li>
            <li>Recall</li>
            <li>F1-Score</li>
            <li>ROC-AUC</li>
        </ul>
    </li>
</ul>

<h2>Key Objectives:</h2>
<ul>
    <li>Preprocess and clean the dataset to handle missing values and normalize features.</li>
    <li>Explore feature importance using techniques like SHAP (SHapley Additive exPlanations).</li>
    <li>Apply machine learning algorithms to predict the presence of heart disease.</li>
    <li>Compare models based on performance metrics to choose the best-fit model for deployment.</li>
</ul>

<h2>Technologies Used:</h2>
<ul>
    <li>Python</li>
    <li>Jupyter Notebook</li>
    <li>Pandas and NumPy for data manipulation</li>
    <li>Scikit-learn for model development and evaluation</li>
    <li>Matplotlib and Seaborn for data visualization</li>
</ul>

<h2>Results:</h2>
<p>The best-performing model is identified based on accuracy, interpretability, and generalization capability. The model can be used as a decision-support tool in clinical settings to aid in heart disease risk prediction.</p>

<h2>Future Work:</h2>
<ul>
    <li>Extend the model to include more features and larger datasets.</li>
    <li>Investigate advanced algorithms like XGBoost or deep learning techniques.</li>
    <li>Integrate the model with a web-based interface for easy accessibility by healthcare professionals.</li>
</ul>
