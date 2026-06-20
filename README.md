# DecodeLabs Tasks

These are my submissions for the DecodeLabs AI Internship (Batch 2026). Each
task builds on the last — starting from plain rule-based logic and working up
to actual machine learning and similarity-based recommendations. I've kept
every project in its own folder with its own README explaining what it does
and why I built it that way.

## Task 1 — Rule-Based AI Chatbot

The first task was about control flow, not AI in the "smart" sense. It's a
chatbot that runs in a loop, cleans up whatever you type (lowercase, strip
whitespace), and looks up a response in a dictionary instead of a long chain
of if-elif statements. If nothing matches, it falls back to a default reply.
Has a couple of small extras too — it can tell you the time and react to "my
name is ___".

## Task 2 — Data Classification Using AI

This one's the actual supervised learning intro. I used the classic Iris
dataset and trained a K-Nearest Neighbors classifier to tell apart the three
flower species from their measurements. The pipeline is: scale the features,
split into train/test sets, fit the model, then check it properly with a
confusion matrix and F1 scores instead of just trusting the accuracy number
(which can be misleading, especially on small or imbalanced data).

## Task 3 — AI Recommendation Logic

The last one is a content-based recommender — no neural networks, no user
history, just comparing a person's skills against a set of job roles and
ranking by similarity. I built a small dataset of roles and their associated
skills, turned everything into TF-IDF vectors so rarer/more specific skills
count for more than generic ones, and used cosine similarity to score and
rank matches. It also has a basic fallback for when someone's input doesn't
overlap with anything in the dataset (the "cold start" problem).

## A note on each project

Every project folder has its own code file plus a README that goes into more
detail — what each script actually does, how to run it, and what I'd try next
if I had more time. Felt like the cleanest way to keep things organized as
the internship adds more tasks each week.
