# Book Discovery Recommender System (DSA 4060XA)

An end-to-end book recommendation engine built on the **Book-Crossing** dataset (~1.15M raw interactions). This project evaluates and contrasts three modeling paradigms: **Popularity Baseline**, **Content-Based Filtering (TF-IDF + Cosine Similarity)**, and **Matrix Factorization (TruncatedSVD)**.

---

## 📌 Project Overview & Problem Statement

* **Target Audience:** Readers seeking personalized book recommendations based on historical explicit ratings ($1-10$ scale).
* **Core Objective:** Surface Top-$N$ relevant books to active users while balancing ranking precision, catalog coverage, and popularity bias.
* **Key Challenge:** Extremely sparse user-item interaction matrix ($>99.9\%$ empty entries), requiring techniques that go beyond naive co-occurrence matching.

---

## 📊 Dataset Statistics & Preprocessing

The primary dataset originates from the **Book-Crossing** explicit rating dump.

| Statistic | Raw Dataset | Cleaned & Filtered Dataset |
| :--- | :--- | :--- |
| **Total Interactions** | 1,149,780 | **145,890** |
| **Unique Users** | 278,858 | **12,293** (users with $\ge 5$ explicit ratings) |
| **Unique Books** | 271,379 | **13,769** (books with $\ge 5$ explicit ratings) |
| **Feedback Signal** | Implicit ($0$) & Explicit ($1-10$) | **Explicit Only ($1-10$)** |
| **Matrix Sparsity** | $99.998\%$ | **$99.91\%$** |

### Data Cleaning Steps:
1. Filtered out implicit $0$ ratings to retain genuine preference signals ($1-10$).
2. Matched ISBN identifiers across `BX-Books.csv` and `BX-Book-Ratings.csv`.
3. Applied iterative core-filtering ($\ge 5$ ratings per user and per item) to ensure dense test evaluation slices.
4. Engineered a composite `metadata` text string (`title` + `author` + `publisher` + `year`) for content representation.

---

## 🏗️ Model Architecture & Pipeline

1. **Popularity Baseline:** Ranks books globally by interaction count (`pop_count`) and average score (`avg_rating`).
2. **Content-Based Filtering:** Vectorizes metadata using TF-IDF (`max_features=5000`, English stop-word removal). Builds user profile vectors weighted by liked books ($\ge 7/10$) and computes Cosine Similarity across unseen candidate items.
3. **Advanced Model (TruncatedSVD):** Reconstructs the user-item preference matrix using $k=20$ latent factors. Centers user ratings by subtracting user means to account for individual grading bias.

---

## 📈 Evaluation & Model Comparison

Models were evaluated on a per-user held-out test split ($20\%$ test ratio, relevance threshold $\ge 7.0$).

| Model | Precision@5 | Recall@5 | Catalog Coverage | Popularity Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | $0.0048$ | $0.0121$ | $0.07\%$ ($0.0007$) | $0.9998$ |
| **Content-Based Filtering** | **$0.0180$** | **$0.0458$** | **$63.10\%$** ($0.6310$) | **$0.6504$** |
| **TruncatedSVD (Latent)** | $0.0038$ | $0.0080$ | $3.04\%$ ($0.0304$) | $0.9068$ |

### Key Findings:
* **Content-Based** achieved the highest Precision@5 ($0.0180$) and catalog coverage ($63.10\%$). On sparse text metadata, matching explicit titles and authors surfaces relevant items effectively.
* **TruncatedSVD** captures individual user taste nuances (e.g., crime mystery series for User `100053` vs. fantasy epics for User `100004`), but suffers from popularity shrinkage on $99.91\%$ sparse rating matrices.
* **Popularity Baseline** suffers from severe popularity bias ($0.9998$), surfacing widely rated but poorly rated books (e.g., *Wild Animus*, avg $3.9/10$) purely due to interaction count.

---

## 🚀 Interactive Streamlit Demonstration

A lightweight user-facing prototype is available in `app/app.py`.

### Running the App:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app/app.py
```

### Prototype Features:
* Real-time User ID selection (`100004`, `100009`, `100053`, etc.).
* Dynamic model switching (**Popularity** vs. **Content-Based** vs. **SVD**).
* Interactive Top-$N$ slider ($3$ to $10$ items).
* Transparent, explainable recommendation reasons for every surfaced item.

---

## 📂 Repository Structure

```
DSA4060XA-Book-Recommender/
├── README.md                           # Project documentation & summary
├── PROJECT_REPORT.pdf                  # Full 10-page final academic report
├── requirements.txt                    # Dependencies (pandas, scikit-learn, streamlit)
├── Book_recommender_system.ipynb       # Complete execution notebook
├── app/
│   └── app.py                          # Streamlit interactive prototype
└── data/
    └── raw/
        ├── BX-Users.csv
        ├── BX-Books.csv
        └── BX-Book-Ratings.csv
```

---
