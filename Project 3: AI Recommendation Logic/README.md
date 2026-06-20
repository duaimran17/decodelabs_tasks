# Project 3: AI Recommendation Logic

This one builds a tech stack recommender — you give it a few skills, and it
tells you which job roles fit best. No neural networks, no historical user
data, just content-based filtering: compare what you know against what each
role needs, and rank by similarity.

## What it does

You type in at least 3 skills (comma-separated), and it returns the top 3
job roles that match. Behind the scenes, every role in `raw_skills.csv` is
turned into a TF-IDF vector, your input gets turned into one too, and the
two get compared with cosine similarity. Whichever roles score highest win.

If your skills don't overlap with anything in the dataset (try typing
"unicorn, magic, fairy dust"), it falls back to a small list of generally
popular roles instead of returning nothing — that's the cold start problem
the slides talked about, handled with a trending fallback.

## Why TF-IDF and not just counting overlapping skills

A plain "does this skill appear, yes/no" comparison treats `python` (which
shows up in half the dataset) the same as `kubernetes` (which only a couple
roles list). That's not how specialization actually works — sharing a rare,
specific skill with a role should count for more than sharing a generic one.
TF-IDF handles that by weighting skills down the more often they appear
across the whole dataset.

## Why cosine similarity and not just distance

Roles with longer skill lists would look artificially "far" from a user
under plain distance metrics, even with a perfect overlap on what they do
share. Cosine similarity looks at the angle between vectors instead of raw
distance, so it's about *proportional* overlap, not how many tags a role
happens to list.

## Files

- `raw_skills.csv` — 14 job roles, each with a list of associated skills
- `recommender.py` — the actual pipeline: load data → vectorize → score → rank → recommend

## Running it

```bash
pip install scikit-learn
python3 recommender.py
```

Then just type skills separated by commas:

```
Your skills: python, sql, cloud computing

Top 3 recommended career paths:
  1. Data Engineer   (match score: 0.55)
  2. Cloud Architect (match score: 0.43)
  3. Data Scientist  (match score: 0.38)
```

Type `exit` to quit.

## Things worth trying

- Add more roles or skills to `raw_skills.csv` and see how the rankings shift
- Swap cosine similarity for Jaccard similarity and compare the results
- Try skill combinations that don't cleanly fit one role and see how it splits the score
