# EasySurv: Advanced Clinical Survival Analysis Tool
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://easysurv.streamlit.app/)

A comprehensive, code-free platform for performing publication-ready survival analysis. Designed for clinicians and researchers, EasySurv takes you from raw data to statistical insight with an integrated **AI Result Narrator**.

![App Screenshot](app_screenshot.png)

## 🌟 Key Features

### 1. 🔍 Data Discovery & Cleanup
*   **Demo Mode**: New to the tool? Click **"Load Demo Data"** on the welcome screen to instantly explore all features with a pre-loaded clinical dataset.
*   **Variable Generation**: 
    *   Create **Combination Variables** (e.g., `TP53_Mut` + `Complex_Karyotype` → 4 groups).
    *   Create **Custom Logic** groups (e.g., `Age > 60` AND `Risk == "High"`).
*   **Correlation Heatmaps**: Instantly visualize relationships between clinical variables using Pearson, Spearman, or Kendall correlations.

### 2. 📊 Interactive Survival Plots (Kaplan-Meier)
*   **Publication-Ready**: 
    *   Clean designs with "Journal" (Nature, Cell, Blood) themes.
    *   Perfectly aligned **Risk Tables** with custom intervals.
    *   **High-Res Downloads**: Export plots at **300 & 600 DPI** (PNG/PDF).
*   **Statistical Tests**: Log-Rank test, Gehan-Breslow-Wilcoxon, and Median Survival estimations.

### 3. 🤖 AI Result Narrator ("Idea-to-Insight")
*   **Auto-Generated Text**: Clicking a single button generates a natural language summary of your results.
    *   *“Group A has significantly inferior survival compared to Group B (p=0.003)...”*
*   **Smart Interpretation**: Context-aware narration of Hazard Ratios, Confidence Intervals, and p-values for Univariable, Multivariable, and CIF analyses.

### 4. 📉 Advanced Statistics
*   **Multivariable Cox Regression**:
    *   Adjust for confounders (Age, Sex, etc.).
    *   **Forest Plots**: Visualize Hazard Ratios with 95% CIs.
    *   **Session Persistence**: Analysis results stay visible while you explore.
    *   🛡️ **Statistical Guardrails (New)**: Automatic warnings for:
        *   **Events Per Variable (EPV)**: Ensures sufficient sample size (using Degrees of Freedom).
        *   **Multicollinearity**: Smart detection for Numeric & Categorical overlaps (VIF + Heatmap).
        *   **Separation**: Detects perfect prediction (infinite HRs).
    *   **Penalized Cox (Ridge)**: Optional toggle to salvage models with severe collinearity or small sample sizes.
*   **Competing Risks Analysis (CIF)**:
    *   **Fine-Gray Regression**: Subdistribution Hazard Ratios for competing events (e.g., Relapse vs Death).
    *   **Cumulative Incidence Plots**: Estimate event probability over time.

### 5. 🧬 Biomarker Optimization
*   **Cutoff Finder**: Automatically scan continuous variables (e.g., Gene Expression) to find the optimal split point that maximizes survival difference (Log-Rank).
*   **Time-Dependent ROC**: Calculate AUC and optimal cutoffs for predicting events at specific time points (e.g., 2-year survival).

### 6. 📄 Reporting
*   **One-Click Report**: Generate a printer-friendly HTML report summarizing all your analyses, plots, and dataset statistics.

### 7. Diagnostic & Prognostic Module (v2.0)
*   **Diagnostic Metrics**: Calculate Sensitivity, Specificity, PPV, NPV, and Accuracy with **95% Wilson Score Confidence Intervals**.

*   **Prognostic "Ladder"**: Compare the Harrell's C-Index of up to **3 models** (e.g., Clinical vs +Biomarker vs +Combined).
*   **Forest Plot**: Visualize the incremental value of biomarkers using a publication-ready Forest Plot with **Bootstrapped 95% CIs** (n=50, Normal Approximation).
*   **Theme Integration**: All plots automatically adapt to your chosen color theme (e.g., NEJM, Nature, or Custom).

### 8. Landmark & Zoom Analysis
*   **Global Landmark Analysis**: Perform conditional survival analysis (e.g., "Survival among patients who were alive at 6 months").
    *   Filters the **entire dataset** (Median, Cox, CIF, Biomarkers) to exclude early events/censors.
    *   Automatically resets Time Zero (`Time = Time - Landmark`).
*   **Zoom Plotting**: Manually focus on a specific time window (e.g., first 24 months) without altering the underlying statistics.

---

## 🚀 How to Run

### Option 1: Web (Streamlit Cloud)
Access the tool directly via the web link (if deployed):
> **[Use the App Here](https://easysurv.streamlit.app/)**

### Option 2: Run Locally (Python)
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/gauravchatnobel/Easysurv.git
    cd Easysurv
    ```

2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## 📂 Demo Data
A dummy dataset `dummy_clinical_data.csv` is included in this repository. 
Columns expected:
*   `Time`: Time to event or censorship.
*   `b`: Binary status (1=Event, 0=Censored).
*   Clinical variables (Age, Gender, etc.).

## 🏗️ Project Structure
The codebase uses a modular architecture for maintainability:
*   `app.py`: Main Streamlit application (UI Layout).
*   `modules/`:
    *   `statistics.py`: Core mathematical functions (Cox, Kaplan-Meier, etc.).
    *   `plotting.py`: Visualization logic.
    *   `narrator.py`: AI Text Generation templates.
    *   `utils.py`: Helper functions, constants, and themes.

## 📄 License
[MIT](LICENSE)
