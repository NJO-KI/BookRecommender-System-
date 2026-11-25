from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender Demo (DSA 4060XA)", layout="wide")

BASE_DIR = Path(
    r"C:\Users\user\OneDrive - United States International University (USIU)\Desktop\FS2026\DSA4060-Austin Odera(Recommender Systems)\Book recomender system"
)


def load_bx_csv(path: Path) -> pd.DataFrame:
    for sep in [";", ",", "\t"]:
        for enc in ["latin1", "utf-8", "cp1252"]:
            try:
                df = pd.read_csv(
                    path, sep=sep, encoding=enc, low_memory=False, on_bad_lines="skip", escapechar="\\"
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("-", "_").replace(" ", "_").replace('"', "")
        if key in {"user_id", "userid"}:
            rename_map[col] = "user_id"
        elif key in {"isbn", "book_id", "item_id", "itemid"}:
            rename_map[col] = "item_id"
        elif key in {"book_rating", "rating"}:
            rename_map[col] = "rating"
        elif key in {"book_title", "title"}:
            rename_map[col] = "title"
        elif key in {"book_author", "author"}:
            rename_map[col] = "author"
        elif key in {"year_of_publication", "year"}:
            rename_map[col] = "year"
        elif key == "publisher":
            rename_map[col] = "publisher"
    return df.rename(columns=rename_map)


@st.cache_data
def build_system():
    books_raw = normalize_columns(load_bx_csv(BASE_DIR / "BX-Books.csv"))
    ratings_raw = normalize_columns(load_bx_csv(BASE_DIR / "BX-Book-Ratings.csv"))

    ratings_df = ratings_raw[["user_id", "item_id", "rating"]].dropna().copy()
    ratings_df["user_id"] = ratings_df["user_id"].astype(str).str.strip()
    ratings_df["item_id"] = ratings_df["item_id"].astype(str).str.strip()
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["rating"])
    ratings_df = ratings_df[ratings_df["rating"] > 0].drop_duplicates(subset=["user_id", "item_id"])

    books_df = books_raw.drop_duplicates(subset=["item_id"]).copy()
    books_df["item_id"] = books_df["item_id"].astype(str).str.strip()
    for col in ["title", "author", "publisher", "year"]:
        books_df[col] = books_df[col].fillna("").astype(str).str.strip() if col in books_df.columns else ""

    ratings_df = ratings_df[ratings_df["item_id"].isin(set(books_df["item_id"]))]
    active_users = ratings_df["user_id"].value_counts()
    active_items = ratings_df["item_id"].value_counts()
    ratings_df = ratings_df[
        ratings_df["user_id"].isin(active_users[active_users >= 5].index)
        & ratings_df["item_id"].isin(active_items[active_items >= 5].index)
    ].reset_index(drop=True)

    books_df = books_df[books_df["item_id"].isin(set(ratings_df["item_id"]))].reset_index(drop=True)
    books_df["metadata"] = (
        books_df["title"] + " " + books_df["author"] + " " + books_df["publisher"] + " " + books_df["year"]
    ).str.lower()

    pop_stats = (
        ratings_df.groupby("item_id")
        .agg(pop_count=("user_id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    pop_ranked_df = (
        books_df[["item_id", "title", "author"]]
        .merge(pop_stats, on="item_id", how="left")
        .fillna({"pop_count": 0, "avg_rating": 0.0})
        .sort_values(["pop_count", "avg_rating"], ascending=False)
        .reset_index(drop=True)
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books_df["metadata"])

    pivot = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0.0)
    user_means = ratings_df.groupby("user_id")["rating"].mean().reindex(pivot.index).fillna(ratings_df["rating"].mean())
    centered = pivot.sub(user_means, axis=0).where(pivot > 0, 0.0)
    n_comp = min(20, min(centered.shape) - 1) if min(centered.shape) > 1 else 1
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    U = svd.fit_transform(centered)
    pred_matrix = pd.DataFrame(
        np.dot(U, svd.components_) + user_means.to_numpy()[:, None],
        index=pivot.index,
        columns=pivot.columns,
    )

    return ratings_df, books_df, pop_ranked_df, tfidf_matrix, pred_matrix


ratings_df, books_df, pop_ranked_df, tfidf_matrix, pred_matrix = build_system()
item_to_title = dict(zip(books_df["item_id"], books_df["title"]))
item_to_idx = {iid: i for i, iid in enumerate(books_df["item_id"])}
idx_to_item = {i: iid for iid, i in item_to_idx.items()}

st.title("Book Recommender System Demo (DSA 4060XA)")
st.caption(
    f"Dataset: {ratings_df['user_id'].nunique():,} users | {books_df['item_id'].nunique():,} books | "
    f"{len(ratings_df):,} explicit ratings"
)

user_options = ["100004", "100009", "100053"] + [
    u for u in ratings_df["user_id"].unique()[:25] if u not in {"100004", "100009", "100053"}
]
selected_user = st.selectbox("Choose Example User ID", user_options)
model_choice = st.radio("Select Recommender Model", ["Popularity", "Content-Based", "SVD"], horizontal=True)
top_k = st.slider("Top-N Recommendations", min_value=3, max_value=10, value=5)

user_hist = ratings_df[ratings_df["user_id"] == selected_user]
seen_items = set(user_hist["item_id"])

col1, col2 = st.columns([1, 1.5])
with col1:
    st.subheader(f"User {selected_user} Reading History ({len(user_hist)} rated)")
    hist_view = user_hist.merge(books_df[["item_id", "title", "author"]], on="item_id", how="left")
    st.dataframe(hist_view[["title", "author", "rating"]].sort_values("rating", ascending=False), hide_index=True)

with col2:
    st.subheader(f"Top-{top_k} Recommendations ({model_choice})")
    recs = []
    if model_choice == "Popularity":
        for _, row in pop_ranked_df.iterrows():
            if row["item_id"] not in seen_items:
                recs.append((row["title"], row["item_id"], f"{int(row['pop_count'])} ratings, avg {row['avg_rating']:.1f}/10"))
            if len(recs) == top_k:
                break
    elif model_choice == "Content-Based":
        liked = user_hist[user_hist["rating"] >= 7.0]
        if liked.empty:
            liked = user_hist
        profile = np.zeros((1, tfidf_matrix.shape[1]), dtype=float)
        anchors = []
        for _, r in liked.iterrows():
            if r["item_id"] in item_to_idx:
                profile += max(float(r["rating"]), 1.0) * tfidf_matrix[item_to_idx[r["item_id"]]].toarray()
                anchors.append(item_to_title.get(r["item_id"], r["item_id"]))
        sims = cosine_similarity(profile, tfidf_matrix).flatten()
        anchor_name = anchors[0] if anchors else "reading history"
        for idx in np.argsort(-sims):
            iid = idx_to_item[idx]
            if iid not in seen_items:
                recs.append((item_to_title.get(iid, iid), iid, f"Similar metadata to '{anchor_name}' (sim={sims[idx]:.2f})"))
            if len(recs) == top_k:
                break
    else:
        if selected_user in pred_matrix.index:
            for iid, score in pred_matrix.loc[selected_user].sort_values(ascending=False).items():
                if iid not in seen_items and iid in item_to_title:
                    recs.append((item_to_title[iid], iid, f"Predicted SVD score: {score:.2f}/10"))
                if len(recs) == top_k:
                    break

    for rank, (title, iid, explanation) in enumerate(recs, start=1):
        st.markdown(f"**{rank}. {title}** (`{iid}`) — {explanation}")