"""
Project 3: AI Recommendation Logic - Tech Stack Recommender
DecodeLabs Industrial Training Kit - Batch 2026

Goal:
    Build a content-based recommendation engine that maps a user's raw
    skills to the job roles ("items") that best match them, using pure
    similarity logic - no historical user data required.

Architecture (IPO Model):
    INPUT   -> User skills (3+) + raw_skills.csv (job roles & their tags)
    PROCESS -> TF-IDF vectorization + Cosine Similarity scoring
    OUTPUT  -> Top-3 ranked job role recommendations

Why content-based filtering?
    Collaborative filtering needs a history of "users who liked X also
    liked Y". We don't have that yet, so this project filters on the
    actual content (skills) of each role instead - which also means
    new roles can be recommended immediately (no item cold-start).

Why TF-IDF instead of plain binary overlap?
    A simple 1/0 "skill present" vector treats a generic skill like
    "python" the same as a specific one like "kubernetes". TF-IDF
    down-weights skills that appear across many roles and up-weights
    skills that are more distinctive - so the similarity math reflects
    genuine specialization, not just raw overlap count.

Why cosine similarity instead of Euclidean distance?
    Euclidean distance is sensitive to how many skills a role lists.
    A role with a long skill list could look "far" from the user purely
    because it has more tags, even if the tags it shares are a perfect
    match. Cosine similarity instead measures the *angle* between the
    user vector and each role vector, so it cares about overlap in
    proportion, not raw magnitude.
"""

import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
TOP_N = 3
MIN_SKILLS = 3

# Used only when a brand-new user has zero overlap with anything in the
# dataset (the "Cold Start" problem) - we fall back to generally popular,
# beginner-friendly roles instead of returning nothing.
TRENDING_FALLBACK = ["Data Scientist", "Backend Developer", "DevOps Engineer"]


# ---------------------------------------------------------------------------
# PHASE 1: INGESTION - Load the item dataset
# ---------------------------------------------------------------------------
def load_roles(path=DATA_PATH):
    """Read raw_skills.csv into two parallel lists: role names and skill text."""
    roles, skill_docs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roles.append(row["role"])
            skill_docs.append(row["skills"])
    return roles, skill_docs


# ---------------------------------------------------------------------------
# PHASE 2: VECTOR MAPPING - Build a shared TF-IDF vocabulary
# ---------------------------------------------------------------------------
def build_vectorizer(skill_docs):
    """
    Fit a TF-IDF vectorizer on every role's skill text. This builds the
    shared vocabulary space that both job roles and the user's input
    must map into - otherwise the similarity math has nothing to compare.
    """
    vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9+/#.\-]+")
    role_vectors = vectorizer.fit_transform(skill_docs)
    return vectorizer, role_vectors


# ---------------------------------------------------------------------------
# PHASE 3: SCORING - Cosine similarity between user and every role
# ---------------------------------------------------------------------------
def score_roles(vectorizer, role_vectors, roles, user_skills):
    """Turn the user's raw skill list into a vector and score it against every role."""
    user_doc = ", ".join(user_skills)
    user_vector = vectorizer.transform([user_doc])

    scores = cosine_similarity(user_vector, role_vectors)[0]
    ranked = sorted(zip(roles, scores), key=lambda pair: pair[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# PHASE 4: SORTING + FILTERING - Top-N list, with a Cold Start bypass
# ---------------------------------------------------------------------------
def recommend(user_skills, top_n=TOP_N):
    if len(user_skills) < MIN_SKILLS:
        raise ValueError(f"Please provide at least {MIN_SKILLS} skills.")

    roles, skill_docs = load_roles()
    vectorizer, role_vectors = build_vectorizer(skill_docs)
    ranked = score_roles(vectorizer, role_vectors, roles, user_skills)

    top_matches = ranked[:top_n]

    # Cold Start bypass: if even the best match has ~zero similarity,
    # the user's skills shared nothing with our vocabulary - fall back
    # to a trending/popular list instead of showing a 0.00 score.
    if top_matches[0][1] < 0.05:
        return [(role, None) for role in TRENDING_FALLBACK[:top_n]]

    return top_matches


def print_recommendations(user_skills, results):
    print("=" * 55)
    print(" PROJECT 3: AI RECOMMENDATION LOGIC - TECH STACK MATCH")
    print("=" * 55)
    print(f"\nYour skills: {', '.join(user_skills)}\n")
    print(f"Top {len(results)} recommended career paths:\n")
    for rank, (role, score) in enumerate(results, start=1):
        if score is None:
            print(f"  {rank}. {role}  (trending fallback - no strong skill overlap found)")
        else:
            print(f"  {rank}. {role}  (match score: {score:.2f})")


if __name__ == "__main__":
    print("Tech Stack Recommender - Project 3")
    print(f"Enter at least {MIN_SKILLS} skills, separated by commas.")
    print("Example: python, sql, cloud computing")
    print("(type 'exit' to quit)\n")

    while True:
        raw = input("Your skills: ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        user_skills = [s.strip() for s in raw.split(",") if s.strip()]

        try:
            results = recommend(user_skills)
        except ValueError as e:
            print(f"\n{e}\n")
            continue

        print()
        print_recommendations(user_skills, results)
        print()
