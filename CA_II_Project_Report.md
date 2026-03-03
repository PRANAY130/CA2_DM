# Project Report: Data Mining and Data Preprocessing for Knowledge Discovery

## 1. Introduction
**Source of the dataset:** 
The dataset implemented and rigorously evaluated in this project is the widely recognized Heart Disease Dataset. It is publicly sourced from the UCI Machine Learning Repository, specifically the Cleveland database, which has served as a benchmark for medical predictive modeling for decades. 

**Application domain:** 
The analytical application domain is strictly placed within Healthcare Analytics and Cardiology.

**Knowledge that can be extracted:** 
The knowledge extracted involves recognizing multi-dimensional predictive patterns and associations between non-invasive patient demographics (e.g., age, sex), continuous clinical metabolic metrics, and the ultimate presence or absence of angiographic heart disease. 

**Real-world decisions supported:** 
Assisting attending physicians in evaluating patients rapidly, assigning risk scores based purely on non-invasive metrics without the need for immediate surgical exploratory tests, and isolating resource-intensive procedures for mathematically prioritized patients.

---

## 2. Dataset Description

**Definitions:**
- **Data object:** An abstract entity or singular instance contained within a dataset matrix. In the specific context of this medical domain, a data object translates physically to a single unique patient.
- **Attributes/features:** The statistically distinct characteristics or observable variables that accurately describe each data object (e.g. cholesterol levels, chest pain presence).

**Dataset Representation:**
Mathematically and structurally, the dataset was verified to be a $303 \times 14$ matrix. Each individual patient is evaluated as a row-wise data object vector $X$:
$$X = (x_1, x_2, \dots, x_{14})$$

---

## 3. Data Objects & Attributes

To successfully execute machine learning algorithms, attributes must be strictly mathematically delineated classifications:

| Attribute | Full Clinical Name | Type Classification | Example Values |
| :--- | :--- | :--- | :--- |
| `age` | Age in years | **Numerical (Ratio)** | 63.0, 37.0, 41.0 |
| `sex` | Sex Classification | **Binary (Symmetric)** | 1 (Male), 0 (Female) |
| `cp` | Chest Pain Type | **Nominal** | 1 (typical), 2 (atypical), 3 (non-anginal), 4 (asymptomatic) |
| `trestbps` | Resting Blood Pressure (mm Hg) | **Numerical (Ratio)** | 145.0, 130.0, 120.0 |
| `chol` | Serum Cholesterol (mg/dl) | **Numerical (Ratio)** | 233.0, 250.0, 204.0 |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | **Binary (Asymmetric)** | 1, 0 |
| `restecg` | Resting Electrocardiographic results | **Nominal** | 0, 1, 2 |
| `thalach` | Max Heart Rate Achieved (bpm) | **Numerical (Ratio)** | 150.0, 187.0, 108.0 |
| `exang` | Exercise Induced Angina Presence | **Binary (Asymmetric)** | 1, 0 |
| `oldpeak` | ST depression induced by exercise | **Numerical (Interval)** | 2.3, 1.5, 3.5 |
| `slope` | Slope of Peak Exercise ST Segment | **Ordinal** | 1, 2, 3 |
| `ca` | Number of Major Vessels Colored | **Numerical (Ratio)** | 0.0, 1.0, 2.0, 3.0 |
| `thal` | Thalassemia Presence/Type | **Nominal** | 3, 6, 7 |
| `target` | Angiographic Heart Disease Diagnosis | **Ordinal / Binary** | 0 (no disease), 1, 2, 3, 4 |

---

## 4. Exploratory Data Analysis (EDA)

Upon execution of Exploratory Data Analysis, the analytical script returned the following concrete numerical results determining our operational parameters:

**Dataset Shape:** Our script output physically verified a structural shape of `(303, 14)`, dictating 303 total patients measured exclusively across 14 vectors.

**Summary Statistics Breakdown:**
An assessment of the raw outputs yields crucial central boundaries:
- **Age:** Spans from a minimum of 29 years to a maximum of 77 years, reflecting an average patient age of precisely `54.43` (${\pm} 9.03$ SD).
- **Cholesterol (`chol`):** Averaging `246.69` mg/dl, notably reaching hazardous maximum tiers nearing `564.0` mg/dl.
- **Maximum Heart Rate (`thalach`):** Outputs logged operational extremes spanning between low bounds of `71.0` up to `202.0` bpm. 

**Which attributes appear important?**
Using extensive correlation analysis and reviewing programmatic histogram distribution density plots, the parameters `cp` (coefficient ~0.43), `thalach` (~-0.42), `exang` (~0.43), `oldpeak` (~0.42), and `ca` (~0.46) demonstrate statistically dominant correlation bounds aligned with diagnosis probability.

**Are there redundant features?**
Based strictly on the rendered heatmaps, pair-wise relationships mostly sit securely between `-0.4` and `0.4`. The greatest inverse redundancy correlation observed mathematically registers specifically between age and max heart rate (`r = -0.393806`). No redundant axes require categorical deletion.

**Are relationships linear or nonlinear?**
While variables like `oldpeak` display rough linear trajectories, graphical densities strictly mapping heart thresholds dictate heavily **nonlinear** relationships demanding computational combinations (like Logistic Regressions) instead of pure linear intercepts.

---

## 5. Similarity and Association Analysis

**Task A: Computed Correlation Outcomes**
Our program extracted the following fundamental matrix snippets evaluating linear (Pearson) vs ranked (Spearman) variances:

*Pearson Snippet Output (Partial):*
```text
               age   trestbps      chol   thalach 
age       1.000000   0.284946  0.208950 -0.393806 
trestbps  0.284946   1.000000  0.130120 -0.045351 
```

*Spearman Snippet Output (Partial):*
```text
               age   trestbps      chol   thalach 
age       1.000000   0.285617  0.195786 -0.380436 
trestbps  0.285617   1.000000  0.126569 -0.044761 
```
*Interpretation:* While both metrics identify severe inverse linkages across age against heart rate (~-0.38), Spearman accurately ignores massive outlier distances, focusing organically purely upon monotonic tracking trajectories without strictly linear proportionality logic.

**Task B: Similarity Comparisons (10-Patient Model)**
Isolating standard features locally, matrix outputs calculated:
- **Euclidean Separation Distance:** Maps literal point-to-point numerical severity. The gap between `patient 0` and `patient 1` output vastly large magnitudes due to raw differences between `thalach` points of 150 vs 108.
- **Cosine Trajectory:** Matrix logs revealed high similarities (${\approx} 0.98+$) tracking closely similar directional clusters despite distinct individual variance heights.
- **Jaccard Mapping:** Operating upon asymmetric arrays `[sex, fbs, exang]`, index returns calculated logical intersections (e.g., patient arrays matching 1s and 0s perfectly mapped Jaccard thresholds of $1.0$).

---

## 6. Data Quality

Null-identification program processes mathematically flagged exact inconsistencies:
```text
Missing Values:
ca       4
thal     2
```
Of over 4,200 absolute datapoints, strictly 6 attributes produced missing outputs. The distribution visually scattered across indices randomly. Because this noise represents approximately $~0.1\%$, dropping the rows truncates $~2\%$ of usable testing data, mandating strict targeted imputations to recover analytical structural integrity successfully.

---

## 7. Intelligent Preprocessing Algorithms & Missing Value Handling

Based precisely on the analysis metrics gathered, applying the mathematical treatments achieved varied outcomes:
| Imputation Method | Effect on Matrix Dynamics | Primary Advantages | Functional Issues |
| :--- | :--- | :--- | :--- |
| **Median Array Imputation** | Instantly swaps the 4 gaps in `ca` arrays dynamically with exactly value `0.0`. | Instant computational speed at zero operational cost. | Strictly compresses dataset variance ignoring underlying multivariable regressions. |
| **KNN (5-Neighbors) Imputation** | Locates $k=5$ identical neighboring patients dynamically, substituting missing parameters algorithmically. | Preserves true metric localization naturally relative to surrounding data groups. | Computationally intense across immense high-dimensional un-scaled subsets. |
| **Iterative Modeling** | Uses random-state Bayesian regression loops strictly mapping available columns to predict absent target gaps logically. | Unparalleled preservation ensuring peak correlation geometry confidence logic. | Demands highly structured distribution thresholds prior to algorithm execution. |

### Validating Preprocessing Pipeline Architectures

Following an exact statistical configuration model mapping `SimpleImputer(median) -> StandardScaler() -> LogisticRegression()`, output splits validated successfully with a standard 80/20 train/test block partition. 

*Analysis Result Log:*
The optimized encapsulation model directly returned a **Test Accuracy: 0.8852 (88.52%)** effectively predicting correct cardiological failure rates securely above standard heuristics.

**Explanation Contexts:**
- **Why pipelines actively matter:** Integrating transformations mechanically into uniform pipeline bounds immediately standardizes reproducibility and deployment scale onto un-flagged metrics seamlessly.
- **Halting Data Leakage:** Executing scaling blindly averages test pools effectively overlapping "unseen" parameters erroneously. Using cross-validated Pipeline shells explicitly locks the `.fit_transform()` behaviors inherently to training architectures alone, generating robust scaling strictly un-influenced by holding behaviors.

---

## 8. Final Knowledge Diagnostics

Reflecting closely upon analytical validation outcomes generated across this assignment:

1. **What physical mathematical patterns were discovered?**
The pipeline probability predictions isolated substantial multi-dimensional boundaries: Classifications operating inside low optimal heart boundaries (`thalach` < 130), severely depressed ST segment lines (`oldpeak` > 2.0), and pronounced asymptotic chest identifiers historically mathematically collapsed exclusively within disease cohorts. 

2. **Which operational attributes generated maximum significance?**
Rigorous correlation extractions heavily weighed baseline parameters:
- `ca` (0.46 correlation coefficient)
- `oldpeak` (0.42 correlation coefficient)
- `cp` and `thalach` components.
Collectively dictating primary Logistic Regression vectors mathematically prioritizing these criteria over standardized bounds like blood sugars.

3. **How specifically did normalization change models?**
Before applying pipeline standard bounds natively, the `chol` parameter randomly commanded regression weights due exclusively to spanning heights of ~560 natively while `oldpeak` stopped precisely at 6.2. `StandardScaler` mechanisms compressed models to balanced standard deviations identical across $0$-mean centers safely preventing metrics with heavily skewed natural integers from unilaterally drowning critical data parameters organically.

4. **What clinical decisions are functionally supported here?**
A rigorously mapped predictive pipeline reliably producing $88.5\%+$ validation scores effectively functions dynamically alongside active working physicians. It actively automates preliminary diagnosis evaluations during saturated triage operations, directly prioritizes limited, invasive angiograms for algorithmically confirmed cases, and dynamically secures computationally reliable preventative diagnostics seamlessly for high-risk flags globally.
