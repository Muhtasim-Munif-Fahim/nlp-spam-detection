"""Synthetic spam/ham text data generation."""

from __future__ import annotations

import random
import pandas as pd


HAM_SAMPLES = [
    "Hey, are we still meeting for lunch tomorrow?",
    "The report is due on Friday, please review.",
    "Can you pick up some milk on your way home?",
    "Thanks for your help with the project.",
    "Don't forget to bring the documents to the meeting.",
    "Your appointment is confirmed for Monday at 3 PM.",
    "Happy birthday! Hope you have a great day!",
    "The server maintenance is scheduled for tonight.",
    "Please review the attached proposal and send feedback.",
    "Meeting notes from today's session are now available.",
]

SPAM_SAMPLES = [
    "CONGRATULATIONS! You've won a FREE iPhone! Click here to claim now!",
    "URGENT: Your account has been compromised. Verify immediately: http://bit.ly/scam",
    "Limited time offer! Buy now and get 50% off! Don't miss out!",
    "You have been selected for a $1000 gift card. Claim at http://spam.com",
    "Win big money now! Casino online! Click here to start winning!",
    "Your Netflix account is on hold. Update payment: http://fake-netflix.com",
    "Earn $5000 per week working from home! No experience needed!",
    "Congratulations! You are the lucky winner of our lottery! Send $50 to claim.",
    "Cheap medications! No prescription needed! Order now and save!",
    "ACT NOW! This offer expires tonight! http://bit.ly/scam-link",
]


def generate_spam_dataset(
    n_ham: int = 100,
    n_spam: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    random.seed(random_state)
    texts = []
    labels = []

    for _ in range(n_ham):
        text = random.choice(HAM_SAMPLES)
        words = text.split()
        random.shuffle(words)
        texts.append(" ".join(words))
        labels.append(0)

    for _ in range(n_spam):
        base = random.choice(SPAM_SAMPLES)
        texts.append(base.upper() if random.random() > 0.5 else base)
        labels.append(1)

    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
