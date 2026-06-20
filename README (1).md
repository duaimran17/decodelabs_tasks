# DecodeLabs Tasks

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Internship-Batch%202026-brightgreen)

My submissions for the DecodeLabs AI Internship (Batch 2026). Three tasks,
three completely different flavors of "intelligence" — and honestly that's
what made this fun. Week 1 is a chatbot that doesn't know anything beyond
what I hardcoded. Week 3 is comparing TF-IDF vectors with cosine similarity.
Same internship, very different muscles.

```
rules (if/else)  →  patterns (ML)  →  similarity (recommendations)
   Task 1              Task 2              Task 3
```

## The lineup

| | Task | What it actually does | Core idea |
|---|---|---|---|
| 🤖 | **Rule-Based Chatbot** | Replies to greetings, time, name, etc. in a loop | Dictionary lookup beats a wall of `elif`s |
| 🌸 | **Data Classification** | Tells apart 3 Iris species from petal/sepal measurements | KNN + train/test split + actually checking F1, not just accuracy |
| 🧭 | **Recommendation Logic** | Matches your skills to the closest-fit job role | TF-IDF weighting + cosine similarity, no neural net required |

### Task 1 — Rule-Based AI Chatbot
Not "AI" in the smart sense — this is pure control flow. Input gets
lowercased and stripped, then looked up in a dictionary instead of a long
if-elif ladder (which the training deck made a pretty convincing case
against, performance-wise). Falls back to a default reply when nothing
matches, and exits cleanly on `bye`/`exit`. Threw in a couple of extras for
fun: it tells the time and reacts to "my name is ___".

### Task 2 — Data Classification Using AI
This is where it stops being "logic I wrote" and starts being "logic the
model figured out." Trained a K-Nearest Neighbors classifier on the Iris
dataset — scale the features, split into train/test, fit, predict. The part
I actually paid attention to was validation: accuracy alone can lie to you,
so I pulled the confusion matrix and per-class F1 scores too, just to be sure
the 100% accuracy wasn't a fluke (Iris is a clean dataset, so it checks out).

### Task 3 — AI Recommendation Logic
A content-based recommender, built from scratch — no historical user data,
no collaborative filtering, just comparing what you know against what a
list of job roles needs. Skills get turned into TF-IDF vectors so a rare
skill like `kubernetes` carries more weight than something generic like
`python` that's everywhere in the dataset. Cosine similarity does the actual
ranking. There's also a small fallback for when someone's input doesn't
overlap with anything — instead of returning nothing, it shows trending
roles (the "cold start problem," handled the cheap way).

## Each folder has its own README

Code + a deeper explanation live together in each project's folder — how to
run it, sample output, and a few ideas for what I'd extend next. This file
is just the overview.
