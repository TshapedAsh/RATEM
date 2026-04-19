# Week 3 Concept — Censoring: Why Survival ≠ Regression

Right-censoring changes the statistical target.

A row with `event = 0` does **not** mean the event never happened.
It means the event was not observed before follow-up ended.

That is why survival analysis keeps **time** and **event status** together, and why Kaplan–Meier is the correct descriptive baseline for censored time-to-event data.
