# Week 1 Concept — Dataset as a Contract

Validation is the first statistical safeguard in RATEM.

This module treats the dataset as a **contract** rather than a vague input table.
Each field should have an explicit definition for:

- type
- allowed values
- units
- missingness rules
- temporal logic

The Week 1 artifact intentionally breaks that contract so the validation layer can fail loudly and export readable audit files.
