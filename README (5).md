# 💊 Cheminformatics Drug Discovery Virtual Laboratory

> *From SMILES to drug-likeness: explore, visualize and analyze molecules the way modern drug discovery teams do.*

## 🎯 Aim

To develop a practical understanding of Cheminformatics by using computational tools to represent, visualize, compare and evaluate chemical structures, with a focus on judging whether a molecule could become a drug candidate.

## 📚 Learning Objectives

After completing this laboratory, students will be able to:

- Convert between molecular representations (SMILES, InChI, InChIKey, Molfile/SDF).
- Visualize molecular structures in 2D and interactive 3D.
- Generate 3D conformers and perform basic energy minimization.
- Calculate and interpret physicochemical molecular descriptors.
- Apply Lipinski's Rule of Five and related drug-likeness filters.
- Estimate drug-likeness and synthetic accessibility using QED and SA scores.
- Generate molecular fingerprints (Morgan/ECFP, MACCS) and compare molecules using Tanimoto similarity.
- Perform substructure searches using SMARTS patterns to identify functional groups.
- Analyze multiple molecules at once (batch input) and export results as CSV.
- Understand structure–property relationships and how small structural changes affect molecular behavior.

## 🧬 Molecular Descriptors and Analyses

The application calculates the following, grouped by the objective they support.

**Basic Identity**
- Molecular Formula
- Molecular Weight
- Exact Mass
- Heavy Atom Count

**Physicochemical Properties**
- LogP (lipophilicity)
- Molar Refractivity
- TPSA (Topological Polar Surface Area)
- Formal Charge

**Hydrogen Bonding and Flexibility**
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
- Fraction of sp³ Carbons (Fsp3)

**Ring and Structural Features**
- Ring Count
- Aromatic Ring Count
- Stereocenter Count

**Drug-Likeness and Filters**
- Lipinski's Rule of Five (pass/fail with violation count)
- QED (Quantitative Estimate of Drug-likeness)
- Synthetic Accessibility (SA) Score

**Similarity and Search**
- Morgan (ECFP4) Fingerprint
- MACCS Keys
- Tanimoto Similarity between molecules
- SMARTS Substructure Match

## 🛠️ Installation

Clone or download this project, then install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` for this laboratory contains:

```text
streamlit
pandas
numpy
matplotlib
py3Dmol
rdkit
reportlab
```

Run the application:

```bash
streamlit run app.py
```
