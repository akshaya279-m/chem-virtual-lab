"""
Cheminformatics Drug Discovery Virtual Laboratory
-------------------------------------------------
Developed by:
    Medatati Akshaya      (RA2611030050066)
    K. Rishitha Archana   (RA2611030050020)

Logo: place your logo file in the same folder as this script and name it
logo.png (also accepted: logo.jpg, logo.jpeg, logo.webp, srm_logo.jpeg).
It is shown automatically at the top-left corner of every page.
"""

import os
import sys
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    MACCSkeys,
    QED,
    RDConfig,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

# 3D visualization
import py3Dmol

# PDF
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# PROJECT DETAILS
# ============================================================

APP_TITLE = "Cheminformatics Drug Discovery Virtual Laboratory"

DEVELOPERS = [
    ("Medatati Akshaya", "RA2611030050066"),
    ("K. Rishitha Archana", "RA2611030050020"),
]

INSTITUTION = "SRM Institute of Science and Technology, Tiruchirappalli"
DEPARTMENT = (
    "Faculty of Engineering and Technology | School of Sciences | "
    "Division of Chemistry"
)

LOGO_CANDIDATES = [
    "logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "srm_logo.jpeg",
]

# Theme colours (drug-discovery palette: deep navy + teal + coral)
NAVY = "#0B2545"
TEAL = "#13A89E"
TEAL_DARK = "#0E7C7B"
MINT = "#E6F6F4"
CORAL = "#FF6B6B"


def find_logo():
    """Return the path of the logo file if the user has added one."""
    base = os.path.dirname(os.path.abspath(__file__))
    for name in LOGO_CANDIDATES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💊",
    layout="wide",
)


# ============================================================
# CUSTOM CSS THEME
# ============================================================

st.markdown(
    f"""
<style>

.stApp {{
    background: linear-gradient(180deg, #F4FBFA 0%, #FFFFFF 40%);
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label, .stApp span,
.stApp div[data-testid="stMarkdownContainer"],
.stApp div[data-testid="stMetricValue"],
.stApp div[data-testid="stMetricLabel"] {{
    color: {NAVY};
}}

.stApp h1, .stApp h2, .stApp h3 {{
    color: {TEAL_DARK};
}}

section[data-testid="stSidebar"] {{
    background: {MINT};
    border-right: 3px solid {TEAL};
}}

.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, {TEAL_DARK} 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 14px;
}}

.hero .hero-title {{
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.2;
}}

.hero .hero-sub {{
    color: #CFF3EE !important;
    font-size: 16px;
    margin-top: 6px;
}}

.card {{
    background: #FFFFFF;
    border: 1px solid #CDEBE7;
    border-left: 6px solid {TEAL};
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(11, 37, 69, 0.06);
}}

.dev-card {{
    background: {MINT};
    border: 1px solid {TEAL};
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}}

.badge-pass {{
    background: #D8F5E3; color: #0B6B35 !important; padding: 2px 10px;
    border-radius: 999px; font-weight: 700;
}}

.badge-fail {{
    background: #FFE0E0; color: #A31515 !important; padding: 2px 10px;
    border-radius: 999px; font-weight: 700;
}}

div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #CDEBE7;
    border-top: 4px solid {TEAL};
    border-radius: 10px;
    padding: 10px 12px;
}}

.stButton > button, .stDownloadButton > button {{
    background: {TEAL_DARK};
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}}

.stButton > button:hover, .stDownloadButton > button:hover {{
    background: {CORAL};
    color: #FFFFFF;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SHARED STATE AND EXAMPLE MOLECULES
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}

# Functional groups used for SMARTS substructure searching
FUNCTIONAL_GROUPS = {
    "Hydroxyl (–OH)": "[OX2H]",
    "Carboxylic acid": "C(=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Amine (primary / secondary)": "[NX3;H2,H1;!$(NC=O)]",
    "Carbonyl (C=O)": "[CX3]=[OX1]",
    "Aromatic ring (benzene)": "c1ccccc1",
    "Ether": "[OD2]([#6])[#6]",
    "Halogen": "[F,Cl,Br,I]",
    "Nitro": "[N+](=O)[O-]",
}

MODEL_STYLES = ["Ball and Stick", "Stick", "Space Filling", "Wireframe"]

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CC(=O)Oc1ccccc1C(=O)O"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Aspirin"


def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]


def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule (None if invalid)."""
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


# ============================================================
# SYNTHETIC ACCESSIBILITY SCORE (RDKit Contrib)
# ============================================================

def _load_sa_scorer():
    try:
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer  # type: ignore
        return sascorer
    except Exception:
        return None


SA_SCORER = _load_sa_scorer()


# ============================================================
# BOND INFORMATION (SIGMA / PI)
# ============================================================

def calculate_bond_information(mol):
    """Heavy-atom bonds, sigma bonds (incl. H), pi bonds and totals."""

    heavy_atom_bonds = mol.GetNumBonds()

    kekule_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": sigma_bonds + pi_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# MOLECULAR DESCRIPTORS
# ============================================================

PROPERTY_GROUPS = {
    "Basic Identity": [
        "Molecular Formula", "Molecular Weight", "Exact Molecular Weight",
        "Number of Atoms", "Heavy Atoms",
    ],
    "Physicochemical": [
        "LogP", "Molar Refractivity", "TPSA", "Formal Charge",
    ],
    "H-Bonding and Flexibility": [
        "Hydrogen Bond Donors", "Hydrogen Bond Acceptors",
        "Rotatable Bonds", "Fraction Csp3",
    ],
    "Rings and Stereochemistry": [
        "Ring Count", "Aromatic Rings", "Aliphatic Rings", "Stereocenters",
    ],
    "Bonding": [
        "Heavy Atom Bonds", "Sigma (σ) Bonds", "Pi (π) Bonds",
        "Total Bonds (σ + π)",
    ],
    "Drug-Likeness": [
        "Lipinski Violations", "Lipinski Rule of Five", "Veber Rule",
        "QED", "SA Score",
    ],
}


def calculate_properties(mol):
    """Calculate the complete descriptor set for one molecule."""

    bond_info = calculate_bond_information(mol)

    mw = round(Descriptors.MolWt(mol), 3)
    logp = round(Crippen.MolLogP(mol), 3)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 3)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)

    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    veber_pass = rot <= 10 and tpsa <= 140

    try:
        qed_value = round(QED.qed(mol), 3)
    except Exception:
        qed_value = "N/A"

    if SA_SCORER is not None:
        try:
            sa_value = round(SA_SCORER.calculateScore(mol), 2)
        except Exception:
            sa_value = "N/A"
    else:
        sa_value = "N/A"

    try:
        stereocenters = len(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        )
    except Exception:
        stereocenters = 0

    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": mw,
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),

        "LogP": logp,
        "Molar Refractivity": round(Crippen.MolMR(mol), 3),
        "TPSA": tpsa,
        "Formal Charge": Chem.GetFormalCharge(mol),

        "Hydrogen Bond Donors": hbd,
        "Hydrogen Bond Acceptors": hba,
        "Rotatable Bonds": rot,
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),

        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Stereocenters": stereocenters,

        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],

        "Lipinski Violations": lipinski_violations,
        "Lipinski Rule of Five": "Pass" if lipinski_violations <= 1 else "Fail",
        "Veber Rule": "Pass" if veber_pass else "Fail",
        "QED": qed_value,
        "SA Score": sa_value,
    }


def properties_dataframe(smiles, mol, properties):
    """Grouped property table (with category column) as a DataFrame."""
    rows = [
        ["Identifiers", "Input SMILES", smiles],
        ["Identifiers", "Canonical SMILES", Chem.MolToSmiles(mol)],
    ]
    try:
        rows.append(["Identifiers", "InChIKey", Chem.MolToInchiKey(mol)])
    except Exception:
        pass
    for group, keys in PROPERTY_GROUPS.items():
        for key in keys:
            rows.append([group, key, str(properties[key])])
    return pd.DataFrame(rows, columns=["Category", "Property", "Value"])


def lipinski_table(properties):
    """Rule-by-rule drug-likeness evaluation."""
    rules = [
        ("Lipinski: Molecular Weight", properties["Molecular Weight"], 500),
        ("Lipinski: LogP", properties["LogP"], 5),
        ("Lipinski: H-Bond Donors", properties["Hydrogen Bond Donors"], 5),
        ("Lipinski: H-Bond Acceptors", properties["Hydrogen Bond Acceptors"], 10),
        ("Veber: Rotatable Bonds", properties["Rotatable Bonds"], 10),
        ("Veber: TPSA", properties["TPSA"], 140),
    ]
    rows = []
    for name, value, limit in rules:
        rows.append({
            "Rule": name,
            "Value": value,
            "Limit (≤)": limit,
            "Status": "✅ Pass" if value <= limit else "❌ Violation",
            "% of Limit": round(100 * value / limit, 1) if limit else 0,
        })
    return pd.DataFrame(rows)


# ============================================================
# FINGERPRINTS, SIMILARITY AND SUBSTRUCTURE
# ============================================================

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_fingerprint(mol):
    return _MORGAN_GEN.GetFingerprint(mol)


def maccs_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def tanimoto(fp1, fp2):
    return round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)


def substructure_scan(mol):
    """Check every preset functional group against the molecule."""
    rows = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        count = len(mol.GetSubstructMatches(patt)) if patt else 0
        rows.append({
            "Functional Group": name,
            "SMARTS": smarts,
            "Present": "✅ Yes" if count else "—",
            "Matches": count,
        })
    return pd.DataFrame(rows)


def highlight_image(mol, smarts, size=(650, 430)):
    """2D image with SMARTS matches highlighted. Returns (image, n_matches)."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None, None
    matches = mol.GetSubstructMatches(patt)
    atoms = sorted({idx for match in matches for idx in match})
    img = Draw.MolToImage(mol, size=size, highlightAtoms=atoms)
    return img, len(matches)


# ============================================================
# 2D / 3D STRUCTURE GENERATION
# ============================================================

def generate_2d_image(mol, size=(700, 500)):
    return Draw.MolToImage(mol, size=size)


def generate_3d_molecule(mol):
    """Embed, then optimise (MMFF, falling back to UFF)."""
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42

        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None

        try:
            result = AllChem.MMFFOptimizeMolecule(mol3d)
            if result == -1:
                AllChem.UFFOptimizeMolecule(mol3d)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d)
            except Exception:
                pass

        return mol3d
    except Exception:
        return None


def mmff_energy(mol3d):
    """MMFF94 energy (kcal/mol) of the generated conformer, if available."""
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol3d)
        ff = AllChem.MMFFGetMoleculeForceField(mol3d, props)
        return round(ff.CalcEnergy(), 3)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_3d_block(smiles):
    """Cached 3D generation: returns (molblock, energy) or (None, None)."""
    mol = get_molecule(smiles)
    if mol is None:
        return None, None
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None, None
    return Chem.MolToMolBlock(mol3d), mmff_energy(mol3d)


def display_3d_structure(mol_block, style, show_surface=False):
    view = py3Dmol.view(width=900, height=600)
    view.addModel(mol_block, "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})

    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=620)


def render_3d_section(smiles, mol, key_prefix):
    """Shared 3D block used by the visualization and properties pages."""
    st.subheader("🌐 Interactive 3D Molecular Structure")

    c1, c2 = st.columns([1, 1])
    with c1:
        style = st.selectbox(
            "Select Molecular Model", MODEL_STYLES, key=f"{key_prefix}_style"
        )
    with c2:
        surface = st.checkbox(
            "Show van der Waals Surface", key=f"{key_prefix}_surface"
        )

    state_key = f"{key_prefix}_3d_smiles"

    if st.button("🚀 Generate 3D Structure", key=f"{key_prefix}_generate"):
        st.session_state[state_key] = smiles

    if st.session_state.get(state_key) == smiles:
        with st.spinner("Generating 3D molecular structure..."):
            block, energy = build_3d_block(smiles)

        if block is None:
            st.error("Unable to generate the 3D structure.")
            return

        m1, m2 = st.columns(2)
        m1.metric(
            "MMFF94 Energy (kcal/mol)",
            energy if energy is not None else "N/A",
        )
        m2.metric("Atoms (with H)", Chem.AddHs(mol).GetNumAtoms())

        display_3d_structure(block, style, surface)

        st.download_button(
            "⬇️ Download 3D Structure (MOL file)",
            data=block,
            file_name="molecule_3d.mol",
            mime="chemical/x-mdl-molfile",
            key=f"{key_prefix}_download_mol",
        )


# ============================================================
# UI HELPERS
# ============================================================

def render_header(title, subtitle=None):
    """Top banner. The logo (if available) is shown at the top-left."""
    logo = find_logo()
    col_logo, col_title = st.columns([1, 6])

    with col_logo:
        if logo:
            st.image(logo, width=120)

    with col_title:
        sub = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
        st.markdown(
            f'<div class="hero"><div class="hero-title">{title}</div>{sub}</div>',
            unsafe_allow_html=True,
        )


def molecule_input_block(key_prefix, label="Enter SMILES"):
    """Shared SMILES input with example loader. Returns (smiles, mol)."""
    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(label, key="active_smiles")

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example",
        )
        st.button(
            "Load Example",
            on_click=load_shared_example,
            key=f"load_example_{key_prefix}",
        )

    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation. Please check the structure.")
    else:
        st.success("✅ Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)

    return smiles, mol


# ============================================================
# PROPERTY INTERPRETATION
# ============================================================

STATIC_INTERPRETATIONS = {
    "Molecular Formula": "Elemental composition of the molecule.",
    "Molecular Weight": (
        "Average molecular mass. Size and mass influence absorption, "
        "diffusion and formulation."
    ),
    "Exact Molecular Weight": (
        "Monoisotopic mass; useful when comparing mass-spectrometry data."
    ),
    "Number of Atoms": (
        "Atoms explicitly present in the RDKit graph (implicit hydrogens "
        "are not counted)."
    ),
    "Heavy Atoms": "Non-hydrogen atoms; a simple measure of molecular size.",
    "Molar Refractivity": (
        "Related to molecular volume and polarizability (how the electron "
        "cloud responds to an electric field)."
    ),
    "Formal Charge": (
        "Net formal charge of the structure; charged molecules are usually "
        "more water-soluble but cross membranes less easily."
    ),
    "Heavy Atom Bonds": "Connections between non-hydrogen atoms.",
    "Sigma (σ) Bonds": (
        "Every bonded atom pair has one σ bond (hydrogen bonds included "
        "in this count)."
    ),
    "Pi (π) Bonds": (
        "π bonds arise from multiple and aromatic bonds and relate to "
        "electron delocalization."
    ),
    "Total Bonds (σ + π)": "Sum of the calculated σ and π bond components.",
    "Lipinski Violations": (
        "Number of Rule-of-Five limits exceeded (MW ≤ 500, LogP ≤ 5, "
        "HBD ≤ 5, HBA ≤ 10)."
    ),
}


def get_property_interpretation(name, value):
    """Student-friendly interpretation for every reported property."""

    if name in STATIC_INTERPRETATIONS and STATIC_INTERPRETATIONS[name]:
        return STATIC_INTERPRETATIONS[name]

    if name == "LogP":
        if value < 0:
            return "Negative: strongly hydrophilic, prefers the aqueous phase."
        if value < 1:
            return "Low lipophilicity; prefers polar environments."
        if value < 3:
            return "Moderate lipophilicity: a balance of polar and non-polar character."
        if value <= 5:
            return "High lipophilicity; may limit solubility but favors membrane crossing."
        return "Very high lipophilicity (above the Lipinski limit of 5)."

    if name == "TPSA":
        if value < 60:
            return "Low polar surface; typically good membrane permeability."
        if value <= 140:
            return "Moderate polar surface; within the Veber limit (≤ 140 Å²)."
        return "High polar surface; oral absorption is likely to be reduced."

    if name == "Hydrogen Bond Donors":
        return f"{value} donor site(s); Lipinski limit is 5."

    if name == "Hydrogen Bond Acceptors":
        return f"{value} acceptor site(s); Lipinski limit is 10."

    if name == "Rotatable Bonds":
        if value == 0:
            return "Rigid framework with no freely rotatable bonds."
        return f"{value} rotatable bond(s); Veber limit is 10 for good oral bioavailability."

    if name == "Fraction Csp3":
        if value < 0.25:
            return "Low sp³ fraction: flat, aromatic or unsaturated character."
        if value < 0.60:
            return "Balanced mix of saturated and unsaturated carbons."
        return "High sp³ fraction: saturated, more three-dimensional framework."

    if name == "Ring Count":
        return f"{value} ring(s) contribute to shape and rigidity."

    if name == "Aromatic Rings":
        return f"{value} aromatic ring(s): planar, delocalized π-electron systems."

    if name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s)."

    if name == "Stereocenters":
        if value == 0:
            return "No stereocenters, so no stereoisomers arise from chiral centers."
        return f"{value} stereocenter(s); stereoisomers can differ in biological activity."

    if name == "Lipinski Rule of Five":
        return (
            "Pass means at most one violation; such molecules are more "
            "likely to be orally bioavailable."
            if value == "Pass"
            else "More than one violation: oral bioavailability may be poor."
        )

    if name == "Veber Rule":
        return (
            "Rotatable bonds ≤ 10 and TPSA ≤ 140 Å² predict good oral "
            "bioavailability."
        )

    if name == "QED":
        if value == "N/A":
            return "QED could not be calculated."
        if value >= 0.67:
            return "High drug-likeness (scale 0–1)."
        if value >= 0.34:
            return "Moderate drug-likeness (scale 0–1)."
        return "Low drug-likeness (scale 0–1)."

    if name == "SA Score":
        if value == "N/A":
            return "SA score module is not available in this RDKit installation."
        if value <= 3:
            return "Easy to synthesize (scale 1 easy to 10 very difficult)."
        if value <= 6:
            return "Moderately difficult to synthesize."
        return "Difficult to synthesize."

    return "Additional quantitative descriptor of the molecular structure."


def build_key_interpretation_points(p):
    """Point-wise conclusions for the PDF report."""
    logp = p["LogP"]
    if logp < 0:
        lipo = "more hydrophilic than lipophilic"
    elif logp < 1:
        lipo = "low in lipophilicity"
    elif logp < 3:
        lipo = "moderately lipophilic"
    else:
        lipo = "strongly lipophilic"

    return [
        f"<b>• Identity:</b> formula <b>{p['Molecular Formula']}</b>, "
        f"molecular weight <b>{p['Molecular Weight']}</b> g/mol.",
        f"<b>• Lipophilicity:</b> LogP = <b>{logp}</b>; the molecule is <b>{lipo}</b>.",
        f"<b>• Polarity:</b> TPSA = <b>{p['TPSA']}</b> Å².",
        f"<b>• Hydrogen bonding:</b> <b>{p['Hydrogen Bond Donors']}</b> donor(s) and "
        f"<b>{p['Hydrogen Bond Acceptors']}</b> acceptor(s).",
        f"<b>• Flexibility:</b> {p['Rotatable Bonds']} rotatable bond(s).",
        f"<b>• Rings and stereochemistry:</b> {p['Ring Count']} ring(s) "
        f"({p['Aromatic Rings']} aromatic) and {p['Stereocenters']} stereocenter(s).",
        f"<b>• Bonding:</b> <b>{p['Sigma (σ) Bonds']}</b> σ bonds and "
        f"<b>{p['Pi (π) Bonds']}</b> π bonds.",
        f"<b>• Drug-likeness:</b> Lipinski Rule of Five: <b>{p['Lipinski Rule of Five']}</b> "
        f"({p['Lipinski Violations']} violation(s)); Veber rule: <b>{p['Veber Rule']}</b>; "
        f"QED = <b>{p['QED']}</b>; SA score = <b>{p['SA Score']}</b>.",
        "<b>• Overall:</b> descriptors should be interpreted together with the 2D "
        "connectivity and 3D geometry; they are computational estimates.",
    ]


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def save_2d_structure_png(mol):
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """Clear ball-and-stick projection of the 3D geometry for the PDF."""

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    coords = np.array(
        [
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(n_atoms)
        ],
        dtype=float,
    )

    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF", "O": "#E53935",
        "F": "#65C466", "P": "#F39C12", "S": "#F4D03F", "Cl": "#27AE60",
        "Br": "#8E3B2F", "I": "#8E44AD",
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260, "P": 300,
        "S": 310, "Cl": 310, "Br": 330, "I": 350,
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    for bond in mol3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)
        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0,
                solid_capstyle="round", zorder=1,
            )

    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y, s=sizes.get(symbol, 250), c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222", linewidths=1.0, zorder=3,
        )
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol, ha="center", va="center", fontsize=8,
                fontweight="bold", color="black", zorder=4,
            )

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    pad = 0.22 * max(max(xmax - xmin, 1.0), max(ymax - ymin, 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(
        "3D Molecular Structure (Ball-and-Stick)",
        fontsize=18, fontweight="bold", pad=18,
    )
    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(
        tmp.name, dpi=320, bbox_inches="tight",
        facecolor="white", pad_inches=0.12,
    )
    plt.close(fig)
    return tmp.name


# ============================================================
# PDF REPORT (single or multiple molecules)
# ============================================================

def generate_pdf_report(report_title, student_name, registration_number, records):
    """
    One PDF for every molecule in `records`.
    records: list of (input_smiles, mol, properties)
    """

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    navy = HexColor(NAVY)
    teal = HexColor(TEAL_DARK)

    title_style = ParagraphStyle(
        "T", parent=styles["Title"], fontSize=19, leading=23,
        alignment=TA_CENTER, textColor=navy,
    )
    heading_style = ParagraphStyle(
        "H", parent=styles["Heading1"], fontSize=13, leading=16,
        textColor=teal, spaceBefore=7, spaceAfter=7,
    )
    body_style = ParagraphStyle(
        "B", parent=styles["BodyText"], fontSize=9, leading=12,
        alignment=TA_JUSTIFY,
    )
    small_style = ParagraphStyle(
        "S", parent=styles["BodyText"], fontSize=7.2, leading=8.8,
    )
    header_style = ParagraphStyle(
        "Hd", parent=styles["BodyText"], fontSize=7.5, leading=9,
        textColor=colors.white, alignment=TA_CENTER,
    )

    def styled_table(data, widths, header_color=teal, label_col=False):
        table = Table(data, colWidths=widths, repeatRows=0 if label_col else 1)
        style = [
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]
        if label_col:
            style.append(("BACKGROUND", (0, 0), (0, -1), HexColor(MINT)))
        else:
            style.append(("BACKGROUND", (0, 0), (-1, 0), header_color))
        table.setStyle(TableStyle(style))
        return table

    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("CHEMINFORMATICS DRUG DISCOVERY VIRTUAL LABORATORY", title_style),
        Paragraph(report_title, heading_style),
        Paragraph(
            f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; "
            f"Registration Number: {registration_number or 'Not provided'}",
            body_style,
        ),
        Paragraph(f"Number of molecules analyzed: {len(records)}", body_style),
        Spacer(1, 0.15 * inch),
    ]

    temp_images = []

    for index, (input_smiles, mol, props) in enumerate(records, start=1):
        if index > 1:
            story.append(PageBreak())

        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = "N/A"

        story.append(Paragraph(f"Molecule {index}: Molecular Information", heading_style))
        info = [
            ["Input SMILES", input_smiles],
            ["Canonical SMILES", Chem.MolToSmiles(mol)],
            ["InChIKey", inchikey],
            ["Molecular Formula", props["Molecular Formula"]],
            ["Molecular Weight", f"{props['Molecular Weight']} g/mol"],
        ]
        story += [
            styled_table(info, [1.8 * inch, 4.9 * inch], label_col=True),
            Spacer(1, 0.12 * inch),
        ]

        story.append(Paragraph("Molecular Structures", heading_style))
        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(mol)
        temp_images += [p for p in (img2d, img3d) if p]

        if img2d and img3d:
            structures = Table(
                [[Image(img2d, width=3.05 * inch, height=2.0 * inch),
                  Image(img3d, width=3.05 * inch, height=2.0 * inch)]],
                colWidths=[3.25 * inch, 3.25 * inch],
            )
            structures.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story += [structures, Spacer(1, 0.12 * inch)]
        elif img2d:
            story += [Image(img2d, width=4.8 * inch, height=3.2 * inch),
                      Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Properties and Interpretation", heading_style))
        prop_rows = [[
            Paragraph("<b>Property</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Interpretation</b>", header_style),
        ]]
        for group, keys in PROPERTY_GROUPS.items():
            for key in keys:
                prop_rows.append([
                    Paragraph(key, small_style),
                    Paragraph(str(props[key]), small_style),
                    Paragraph(get_property_interpretation(key, props[key]), small_style),
                ])
        story += [
            styled_table(prop_rows, [1.55 * inch, 0.9 * inch, 4.25 * inch]),
            Spacer(1, 0.12 * inch),
        ]

        story.append(Paragraph("Drug-Likeness Rule Check", heading_style))
        rule_df = lipinski_table(props)
        rule_rows = [[
            Paragraph("<b>Rule</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Limit (≤)</b>", header_style),
            Paragraph("<b>Status</b>", header_style),
        ]]
        for _, row in rule_df.iterrows():
            rule_rows.append([
                Paragraph(str(row["Rule"]), small_style),
                Paragraph(str(row["Value"]), small_style),
                Paragraph(str(row["Limit (≤)"]), small_style),
                Paragraph("Pass" if "Pass" in row["Status"] else "Violation", small_style),
            ])
        story += [
            styled_table(rule_rows, [2.6 * inch, 1.2 * inch, 1.2 * inch, 1.7 * inch]),
            Spacer(1, 0.12 * inch),
        ]

        story.append(Paragraph("Key Interpretation", heading_style))
        for point in build_key_interpretation_points(props):
            story += [Paragraph(point, body_style), Spacer(1, 0.04 * inch)]

    # ---------------- libraries and code ----------------
    story += [PageBreak(), Paragraph("Python Libraries Used and Their Importance", heading_style)]

    library_rows = [
        [Paragraph("<b>Library</b>", header_style),
         Paragraph("<b>Importance in this project</b>", header_style)],
        [Paragraph("Streamlit", small_style),
         Paragraph("Interactive web application, navigation, inputs and downloads.", small_style)],
        [Paragraph("RDKit", small_style),
         Paragraph("Parses SMILES, draws 2D structures, generates 3D conformers, calculates "
                   "descriptors, QED, fingerprints, similarity and SMARTS substructure matches.", small_style)],
        [Paragraph("Pandas", small_style),
         Paragraph("Tables of molecular properties and CSV export.", small_style)],
        [Paragraph("Matplotlib and NumPy", small_style),
         Paragraph("Charts and projection of 3D coordinates for the PDF figures.", small_style)],
        [Paragraph("py3Dmol", small_style),
         Paragraph("Interactive 3D molecular viewer.", small_style)],
        [Paragraph("ReportLab", small_style),
         Paragraph("Builds this downloadable PDF report.", small_style)],
    ]
    story += [
        styled_table(library_rows, [1.8 * inch, 4.9 * inch]),
        Spacer(1, 0.15 * inch),
        Paragraph("Illustrative Python Code", heading_style),
    ]

    code_style = ParagraphStyle(
        "C", parent=styles["Code"], fontName="Courier", fontSize=7,
        leading=8.5, leftIndent=4, rightIndent=4,
    )
    code_example = "\n".join([
        "from rdkit import Chem, DataStructs",
        "from rdkit.Chem import Descriptors, Crippen, QED, rdFingerprintGenerator",
        "",
        'mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin',
        'print("MW  :", Descriptors.MolWt(mol))',
        'print("LogP:", Crippen.MolLogP(mol))',
        'print("QED :", QED.qed(mol))',
        "",
        "# Fingerprint similarity (Tanimoto)",
        "gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)",
        "fp1 = gen.GetFingerprint(mol)",
        'ref = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")  # paracetamol',
        "fp2 = gen.GetFingerprint(ref)",
        'print("Tanimoto:", DataStructs.TanimotoSimilarity(fp1, fp2))',
        "",
        "# SMARTS substructure search (carboxylic acid)",
        'patt = Chem.MolFromSmarts("C(=O)[OX2H1]")',
        'print("Has acid:", mol.HasSubstructMatch(patt))',
    ])
    story += [
        Preformatted(code_example, code_style, maxLineLength=92),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "This compact example shows the core workflow. The full application adds "
            "visualization, interpretation and report generation. Calculated descriptors "
            "are computational estimates and should be interpreted with their definitions.",
            body_style,
        ),
        Spacer(1, 0.12 * inch),
        Paragraph(
            "Developed by: "
            + "; ".join(f"{n} ({r})" for n, r in DEVELOPERS),
            body_style,
        ),
    ]

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for path in temp_images:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💊 Drug Discovery Lab")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "💊 Drug-Likeness Screening",
        "🔍 Similarity & Substructure Search",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input(
    "Registration Number", key="registration_number"
)

st.sidebar.markdown(
    """
### 🎓 Learning Objectives

Students can learn to:

- Convert SMILES, InChI and Molfile
- Visualize molecules in 2D and 3D
- Generate 3D conformers and energies
- Calculate molecular descriptors
- Apply Lipinski and Veber rules
- Estimate QED and SA score
- Compare molecules with fingerprints
- Search substructures with SMARTS
- Analyze batches and export CSV
- Link structure to properties
"""
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    render_header(
        "💊 " + APP_TITLE,
        f"{INSTITUTION}<br>{DEPARTMENT}",
    )

    st.markdown("## Welcome to the Virtual Laboratory")

    st.write(
        "This interactive laboratory introduces cheminformatics through the "
        "lens of drug discovery: represent a molecule, visualize it, measure "
        "its properties and decide whether it behaves like a drug candidate."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### 🧪 What can students do?")
        st.markdown(
            """
✅ Enter SMILES and validate structures

✅ Convert to InChI, InChIKey and Molfile

✅ Generate 2D and interactive 3D structures

✅ Compute MMFF energy of a 3D conformer

✅ Calculate a complete descriptor set

✅ Screen with Lipinski, Veber, QED and SA score

✅ Compare molecules by Tanimoto similarity

✅ Search functional groups with SMARTS

✅ Run batch analyses and download CSV / PDF
"""
        )

    with right:
        st.markdown("### 📋 Descriptors Calculated")
        for group, keys in PROPERTY_GROUPS.items():
            st.markdown(f"**{group}:** " + ", ".join(keys))

    st.markdown("### Example Molecules")
    st.dataframe(
        pd.DataFrame({
            "Molecule": list(EXAMPLE_MOLECULES.keys()),
            "SMILES": list(EXAMPLE_MOLECULES.values()),
        }),
        hide_index=True,
    )

    st.divider()

    st.markdown("### 👩‍💻 Developed by")
    dev_cols = st.columns(len(DEVELOPERS))
    for col, (name, reg) in zip(dev_cols, DEVELOPERS):
        with col:
            st.markdown(
                f'<div class="dev-card"><b>{name}</b><br>'
                f"Registration No.: {reg}<br>{INSTITUTION}</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":

    render_header("📖 Introduction to Cheminformatics")

    st.markdown(
        """
## What is Cheminformatics?

Cheminformatics combines **Chemistry ⚗️**, **Computer Science 💻** and
**Data Analysis 📊** to store, process, visualize and analyze chemical
information. In drug discovery it helps researchers decide which molecules
deserve expensive laboratory testing.

## Molecular Representations

| Format | Purpose | Example (ethanol) |
|---|---|---|
| SMILES | Text line notation of structure | CCO |
| InChI | Standard, unique text identifier | InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3 |
| InChIKey | Fixed-length hash of the InChI | LFQSCWFLJHTTHZ-UHFFFAOYSA-N |
| Molfile / SDF | Atom coordinates and bonds | 2D or 3D coordinate file |

## Molecular Descriptors

Numerical values calculated from a structure: molecular weight, LogP, TPSA,
hydrogen-bond donors and acceptors, rotatable bonds, ring counts, Fsp3,
stereocenters, formal charge and more.

## Drug-Likeness

- **Lipinski's Rule of Five:** MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
  (at most one violation is tolerated).
- **Veber rule:** rotatable bonds ≤ 10 and TPSA ≤ 140 Å².
- **QED:** a 0–1 score combining several properties into one drug-likeness value.
- **SA score:** synthetic accessibility from 1 (easy) to 10 (very difficult).

## Fingerprints and Similarity

A **fingerprint** converts a structure into a bit string. The **Tanimoto
coefficient** (0 to 1) compares two fingerprints: values near 1 mean very
similar structures. Common types: Morgan/ECFP4 and MACCS keys.

## Substructure Search with SMARTS

**SMARTS** is a pattern language for functional groups. For example,
`C(=O)[OX2H1]` finds carboxylic acids.

## 3D Models

- 🟢 **Ball and Stick:** atoms as spheres, bonds as sticks
- 🟡 **Stick:** emphasizes bonding
- 🔵 **Space Filling:** approximate molecular volume
- ⚫ **Wireframe:** simplified skeleton

## Structure–Property Relationships

Small structural changes (adding an –OH, a ring or a halogen) alter LogP,
polarity and drug-likeness. Comparing molecules side by side reveals these
trends.
"""
    )


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":

    render_header("🧬 Molecular Visualization Laboratory")

    st.info(
        "Enter a SMILES or load an example. The selected molecule is shared "
        "across all laboratory pages."
    )

    smiles, mol = molecule_input_block("visualization")

    if mol is not None:

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("2D Molecular Structure")
            st.image(generate_2d_image(mol))

        with col2:
            st.subheader("Identifiers and Counts")
            bond_info = calculate_bond_information(mol)

            st.write("**Input SMILES:**", smiles)
            st.write("**Canonical SMILES:**", Chem.MolToSmiles(mol))
            try:
                st.write("**InChI:**", Chem.MolToInchi(mol))
                st.write("**InChIKey:**", Chem.MolToInchiKey(mol))
            except Exception:
                st.write("**InChI:** not available")
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Heavy Atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Total Atoms (including H):**", bond_info["Total Atoms Including H"])
            st.write("**Sigma (σ) Bonds:**", bond_info["Sigma Bonds"])
            st.write("**Pi (π) Bonds:**", bond_info["Pi Bonds"])

            st.download_button(
                "⬇️ Download 2D Structure (MOL file)",
                data=Chem.MolToMolBlock(mol),
                file_name="molecule_2d.mol",
                mime="chemical/x-mdl-molfile",
                key="download_2d_mol",
            )

        st.divider()
        render_3d_section(smiles, mol, "viz")


# ============================================================
# MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":

    render_header("📊 Complete Molecular Properties Report")

    st.info(
        "Enter any valid SMILES. The laboratory generates the complete "
        "descriptor report, including drug-likeness."
    )

    smiles, mol = molecule_input_block("properties")

    if mol is not None:

        properties = calculate_properties(mol)
        canonical_smiles = Chem.MolToSmiles(mol)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("🧬 2D Molecular Structure")
            st.image(generate_2d_image(mol))

        with col2:
            st.subheader("📋 Basic Information")
            st.write("**Input SMILES:**", smiles)
            st.write("**Canonical SMILES:**", canonical_smiles)
            st.write("**Molecular Formula:**", properties["Molecular Formula"])
            st.write("**Atoms / Heavy Atoms:**",
                     f"{properties['Number of Atoms']} / {properties['Heavy Atoms']}")
            st.write("**σ / π Bonds:**",
                     f"{properties['Sigma (σ) Bonds']} / {properties['Pi (π) Bonds']}")
            st.write("**Stereocenters:**", properties["Stereocenters"])

        st.divider()
        st.subheader("📊 Important Molecular Properties")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Molecular Weight", f"{properties['Molecular Weight']} g/mol")
        c2.metric("LogP", properties["LogP"])
        c3.metric("TPSA", f"{properties['TPSA']} Å²")
        c4.metric("Ring Count", properties["Ring Count"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("H-Bond Donors", properties["Hydrogen Bond Donors"])
        c2.metric("H-Bond Acceptors", properties["Hydrogen Bond Acceptors"])
        c3.metric("Rotatable Bonds", properties["Rotatable Bonds"])
        c4.metric("Fraction Csp3", properties["Fraction Csp3"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lipinski Ro5", properties["Lipinski Rule of Five"])
        c2.metric("Veber Rule", properties["Veber Rule"])
        c3.metric("QED", properties["QED"])
        c4.metric("SA Score", properties["SA Score"])

        st.divider()
        st.subheader("📑 Complete Molecular Property Table")

        report_df = properties_dataframe(smiles, mol, properties)
        st.dataframe(report_df, hide_index=True)

        st.download_button(
            "⬇️ Download Properties Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="Molecular_Properties_Report.csv",
            mime="text/csv",
        )

        if st.button("📄 Generate Molecular PDF Report"):
            with st.spinner("Generating molecular PDF report..."):
                pdf = generate_pdf_report(
                    "MOLECULAR PROPERTIES REPORT",
                    student_name,
                    registration_number,
                    [(smiles, mol, properties)],
                )
            st.download_button(
                "⬇️ Download Molecular Properties PDF",
                data=pdf,
                file_name="Molecular_Properties_Report.pdf",
                mime="application/pdf",
            )

        st.divider()
        render_3d_section(smiles, mol, "props")


# ============================================================
# DRUG-LIKENESS SCREENING
# ============================================================

elif page == "💊 Drug-Likeness Screening":

    render_header(
        "💊 Drug-Likeness Screening",
        "Lipinski Rule of Five · Veber Rule · QED · Synthetic Accessibility",
    )

    smiles, mol = molecule_input_block("druglike")

    if mol is not None:

        props = calculate_properties(mol)
        rules_df = lipinski_table(props)

        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.image(generate_2d_image(mol, size=(500, 380)))

        with col2:
            m1, m2 = st.columns(2)
            m1.metric("Lipinski Violations", props["Lipinski Violations"])
            m2.metric("Lipinski Ro5", props["Lipinski Rule of Five"])
            m3, m4 = st.columns(2)
            m3.metric("QED (0–1)", props["QED"])
            m4.metric("SA Score (1–10)", props["SA Score"])

            if props["Lipinski Rule of Five"] == "Pass" and props["Veber Rule"] == "Pass":
                st.success("✅ Passes Lipinski and Veber: good oral drug-likeness profile.")
            else:
                st.warning(
                    "⚠️ Fails at least one rule set: oral bioavailability may be limited."
                )

        st.subheader("📋 Rule-by-Rule Evaluation")
        st.dataframe(rules_df, hide_index=True)

        st.subheader("📊 Property Value as % of Rule Limit")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        bar_colors = [TEAL if v <= 100 else CORAL for v in rules_df["% of Limit"]]
        ax.barh(rules_df["Rule"], rules_df["% of Limit"], color=bar_colors)
        ax.axvline(100, color=NAVY, linestyle="--", linewidth=1.2, label="Limit (100%)")
        ax.set_xlabel("% of limit")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Bars beyond the dashed line (coral) exceed the rule limit. "
            "Negative LogP values appear as bars below 0%."
        )

        st.download_button(
            "⬇️ Download Drug-Likeness Table (CSV)",
            data=rules_df.to_csv(index=False).encode("utf-8"),
            file_name="drug_likeness_screening.csv",
            mime="text/csv",
        )


# ============================================================
# SIMILARITY AND SUBSTRUCTURE SEARCH
# ============================================================

elif page == "🔍 Similarity & Substructure Search":

    render_header(
        "🔍 Similarity & Substructure Search",
        "Fingerprints · Tanimoto similarity · SMARTS functional-group search",
    )

    tab_fp, tab_sim, tab_sub = st.tabs(
        ["🧬 Fingerprints", "🤝 Tanimoto Similarity", "🔎 SMARTS Substructure Search"]
    )

    # ---------------- FINGERPRINTS ----------------
    with tab_fp:
        smiles_fp, mol_fp = molecule_input_block("fingerprint")

        if mol_fp is not None:
            morgan = morgan_fingerprint(mol_fp)
            maccs = maccs_fingerprint(mol_fp)

            c1, c2 = st.columns(2)
            c1.metric("Morgan (ECFP4, 2048 bits): bits ON", morgan.GetNumOnBits())
            c2.metric("MACCS keys (167 bits): bits ON", maccs.GetNumOnBits())

            st.write("**Morgan bits that are ON:**")
            st.code(", ".join(map(str, list(morgan.GetOnBits()))) or "none")
            st.write("**MACCS keys that are ON:**")
            st.code(", ".join(map(str, list(maccs.GetOnBits()))) or "none")

    # ---------------- SIMILARITY ----------------
    with tab_sim:
        st.write(
            "Compare a **reference** molecule with a list of query molecules. "
            "Enter queries as **Name, SMILES** (one per line)."
        )

        ref_smiles = st.text_input(
            "Reference SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="similarity_ref"
        )
        query_text = st.text_area(
            "Query molecules",
            value=(
                "Paracetamol,CC(=O)Nc1ccc(O)cc1\n"
                "Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O\n"
                "Salicylic acid,OC(=O)c1ccccc1O\n"
                "Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C\n"
                "Ethanol,CCO"
            ),
            height=150,
            key="similarity_queries",
        )

        ref_mol = get_molecule(ref_smiles)

        if ref_mol is None:
            st.error("❌ Invalid reference SMILES.")
        else:
            ref_morgan = morgan_fingerprint(ref_mol)
            ref_maccs = maccs_fingerprint(ref_mol)

            rows = []
            query_mols = {}
            for line in query_text.splitlines():
                if "," not in line:
                    continue
                name, sm = line.split(",", 1)
                qm = get_molecule(sm)
                if qm is None:
                    continue
                query_mols[name.strip()] = qm
                rows.append({
                    "Molecule": name.strip(),
                    "SMILES": sm.strip(),
                    "Tanimoto (Morgan)": tanimoto(ref_morgan, morgan_fingerprint(qm)),
                    "Tanimoto (MACCS)": tanimoto(ref_maccs, maccs_fingerprint(qm)),
                })

            if rows:
                sim_df = pd.DataFrame(rows).sort_values(
                    "Tanimoto (Morgan)", ascending=False
                )
                st.dataframe(sim_df, hide_index=True)

                fig, ax = plt.subplots(figsize=(8, 3.6))
                ax.barh(sim_df["Molecule"], sim_df["Tanimoto (Morgan)"], color=TEAL)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Tanimoto similarity (Morgan)")
                ax.invert_yaxis()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                best = sim_df.iloc[0]["Molecule"]
                grid = Draw.MolsToGridImage(
                    [ref_mol, query_mols[best]],
                    legends=["Reference", f"Most similar: {best}"],
                    subImgSize=(350, 280),
                    molsPerRow=2,
                )
                st.image(grid)

                st.download_button(
                    "⬇️ Download Similarity Table (CSV)",
                    data=sim_df.to_csv(index=False).encode("utf-8"),
                    file_name="similarity_results.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Enter at least one valid query as: Name, SMILES")

    # ---------------- SUBSTRUCTURE ----------------
    with tab_sub:
        smiles_sub, mol_sub = molecule_input_block("substructure")

        if mol_sub is not None:
            st.subheader("Functional-Group Scan")
            st.dataframe(substructure_scan(mol_sub), hide_index=True)

            st.subheader("Highlight a Pattern")
            mode = st.radio(
                "Pattern source",
                ["Preset functional group", "Custom SMARTS"],
                horizontal=True,
                key="smarts_mode",
            )
            if mode == "Preset functional group":
                group = st.selectbox(
                    "Functional group", list(FUNCTIONAL_GROUPS.keys()), key="smarts_group"
                )
                smarts = FUNCTIONAL_GROUPS[group]
                st.code(smarts)
            else:
                smarts = st.text_input("SMARTS pattern", value="c1ccccc1", key="smarts_custom")

            img, n_matches = highlight_image(mol_sub, smarts)
            if img is None:
                st.error("❌ Invalid SMARTS pattern.")
            else:
                st.image(img)
                if n_matches:
                    st.success(f"✅ Pattern found {n_matches} time(s); matches are highlighted.")
                else:
                    st.info("Pattern not found in this molecule.")


# ============================================================
# STRUCTURE PROPERTY ANALYSIS (BATCH)
# ============================================================

elif page == "📈 Structure–Property Analysis":

    render_header("📈 Structure–Property Analysis")

    st.write(
        "Compare the properties of multiple molecules. Enter one molecule "
        "per line in the format:\n\n**Molecule Name, SMILES**"
    )

    default_data = (
        "Ethanol,CCO\n"
        "Benzene,c1ccccc1\n"
        "Phenol,Oc1ccccc1\n"
        "Acetic Acid,CC(=O)O\n"
        "Aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
        "Paracetamol,CC(=O)Nc1ccc(O)cc1\n"
        "Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    )

    molecule_input = st.text_area("Enter Molecules", value=default_data, height=200)

    rows = []
    for line in molecule_input.splitlines():
        if "," not in line:
            continue
        name, sm = line.split(",", 1)
        m = get_molecule(sm)
        if m:
            p = calculate_properties(m)
            rows.append({
                "Molecule": name.strip(),
                "MW": p["Molecular Weight"],
                "LogP": p["LogP"],
                "TPSA": p["TPSA"],
                "HBD": p["Hydrogen Bond Donors"],
                "HBA": p["Hydrogen Bond Acceptors"],
                "Rotatable Bonds": p["Rotatable Bonds"],
                "Ring Count": p["Ring Count"],
                "Fsp3": p["Fraction Csp3"],
                "QED": p["QED"],
                "Lipinski Violations": p["Lipinski Violations"],
                "Lipinski Ro5": p["Lipinski Rule of Five"],
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        st.download_button(
            "⬇️ Download Comparison Table (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="structure_property_comparison.csv",
            mime="text/csv",
            key="download_structure_property_csv",
        )

        st.subheader("📉 Explore a Relationship")
        numeric_cols = [
            "MW", "LogP", "TPSA", "HBD", "HBA", "Rotatable Bonds",
            "Ring Count", "Fsp3", "QED", "Lipinski Violations",
        ]
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X-axis property", numeric_cols, index=0, key="sp_x")
        y_col = c2.selectbox("Y-axis property", numeric_cols, index=1, key="sp_y")

        plot_df = df.copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col, y_col])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(plot_df[x_col], plot_df[y_col], s=90, color=TEAL, edgecolor=NAVY)
        for _, row in plot_df.iterrows():
            ax.annotate(
                row["Molecule"], (row[x_col], row[y_col]),
                textcoords="offset points", xytext=(6, 6), fontsize=8,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Please enter at least one valid molecule in the format: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":

    render_header("📝 Cheminformatics Assessment")

    questions = [
        {"question": "What is the full form of SMILES?", "options": ["Simplified Molecular Input Line Entry System", "Standard Molecular Information Language Encoding System", "Simple Molecular Identification and Labeling System"], "answer": "Simplified Molecular Input Line Entry System"},
        {"question": "Which SMILES represents ethanol?", "options": ["CCO", "c1ccccc1", "CC(=O)O"], "answer": "CCO"},
        {"question": "Which SMILES represents benzene?", "options": ["c1ccccc1", "CCO", "O=C=O"], "answer": "c1ccccc1"},
        {"question": "In SMILES, parentheses are mainly used to indicate:", "options": ["Branches", "Molecular weight", "Atom colour"], "answer": "Branches"},
        {"question": "Which property is commonly associated with lipophilicity?", "options": ["LogP", "TPSA", "HBD"], "answer": "LogP"},
        {"question": "What does TPSA describe?", "options": ["Topological Polar Surface Area", "Total Pi Surface Area", "Thermal Property Surface Analysis"], "answer": "Topological Polar Surface Area"},
        {"question": "A double bond contains how many pi (π) bonds?", "options": ["1", "2", "0"], "answer": "1"},
        {"question": "A triple bond contains how many pi (π) bonds?", "options": ["2", "1", "3"], "answer": "2"},
        {"question": "Which model is useful for visualizing approximate molecular volume?", "options": ["Space Filling", "Wireframe only", "Text SMILES"], "answer": "Space Filling"},
        {"question": "Lipinski's Rule of Five states that molecular weight should be at most:", "options": ["500", "250", "1000"], "answer": "500"},
        {"question": "According to Lipinski's rule, the maximum number of hydrogen-bond donors is:", "options": ["5", "10", "2"], "answer": "5"},
        {"question": "The Veber rule limits TPSA to at most:", "options": ["140 Å²", "40 Å²", "300 Å²"], "answer": "140 Å²"},
        {"question": "The QED score ranges from:", "options": ["0 to 1", "1 to 10", "-5 to 5"], "answer": "0 to 1"},
        {"question": "A synthetic accessibility (SA) score close to 1 means the molecule is:", "options": ["Easy to synthesize", "Very difficult to synthesize", "Toxic"], "answer": "Easy to synthesize"},
        {"question": "A Tanimoto coefficient of 1.0 between two fingerprints means:", "options": ["Identical fingerprints", "Completely different molecules", "Both molecules are drugs"], "answer": "Identical fingerprints"},
        {"question": "SMARTS patterns are mainly used for:", "options": ["Substructure searching", "Calculating molecular weight", "Drawing 3D models"], "answer": "Substructure searching"},
        {"question": "The InChIKey is:", "options": ["A fixed-length hashed identifier of a structure", "A 3D coordinate file", "A fingerprint type"], "answer": "A fixed-length hashed identifier of a structure"},
        {"question": "A high Fraction Csp3 indicates a molecule that is:", "options": ["More saturated and three-dimensional", "Completely aromatic", "Always charged"], "answer": "More saturated and three-dimensional"},
        {"question": "Which fingerprint type is also known as ECFP?", "options": ["Morgan", "MACCS", "InChI"], "answer": "Morgan"},
        {"question": "Increasing the number of –OH groups in a molecule generally:", "options": ["Increases polarity (TPSA) and lowers LogP", "Raises LogP strongly", "Has no effect on any property"], "answer": "Increases polarity (TPSA) and lowers LogP"},
    ]

    student_answers = []
    for i, q in enumerate(questions):
        answer = st.radio(
            f"{i + 1}. {q['question']}",
            q["options"],
            index=None,
            key=f"assessment_v3_{i}",
        )
        student_answers.append(answer)

    if st.button("Submit Assessment"):

        if any(a is None for a in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()

        score = sum(
            1 for i, q in enumerate(questions) if student_answers[i] == q["answer"]
        )
        percentage = score / len(questions) * 100

        st.success(f"Your Score: {score}/{len(questions)}")
        st.metric("Percentage", f"{percentage:.1f}%")

        if percentage >= 80:
            st.balloons()
            st.success("🎉 Excellent performance!")
        elif percentage >= 50:
            st.info("👍 Good! Continue practicing.")
        else:
            st.warning("📚 Please review the Theory section and try again.")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":

    render_header("📄 Final Student Molecular Analysis Report")

    st.write(
        "Enter one SMILES string per line. The report includes every valid "
        "molecule in a single downloadable PDF."
    )

    name = st.text_input(
        "Student Name for Report",
        value=st.session_state.get("student_name", ""),
        key="final_report_name",
    )
    reg = st.text_input(
        "Registration Number for Report",
        value=st.session_state.get("registration_number", ""),
        key="final_report_reg",
    )

    if "final_report_smiles_multi" not in st.session_state:
        st.session_state.final_report_smiles_multi = st.session_state.get(
            "latest_smiles", st.session_state.get("active_smiles", "CCO")
        )

    multi_smiles_text = st.text_area(
        "Enter SMILES (one molecule per line)",
        height=150,
        key="final_report_smiles_multi",
        help="Example: CCO on the first line, c1ccccc1 on the second line, and so on.",
    )

    molecule_records = []
    invalid_smiles = []

    for item in [l.strip() for l in multi_smiles_text.splitlines() if l.strip()]:
        m = get_molecule(item)
        if m is None:
            invalid_smiles.append(item)
        else:
            molecule_records.append((item, m, calculate_properties(m)))

    if invalid_smiles:
        st.warning(
            "These SMILES could not be interpreted and will not be included: "
            + ", ".join(invalid_smiles)
        )

    if molecule_records:
        st.success(f"{len(molecule_records)} valid molecule(s) ready for the report.")

        preview = pd.DataFrame([
            {
                "No.": i,
                "Input SMILES": s,
                "Molecular Formula": p["Molecular Formula"],
                "MW (g/mol)": p["Molecular Weight"],
                "Lipinski Ro5": p["Lipinski Rule of Five"],
                "QED": p["QED"],
            }
            for i, (s, m, p) in enumerate(molecule_records, start=1)
        ])
        st.dataframe(preview, hide_index=True)

        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner(
                f"Generating one PDF report for {len(molecule_records)} molecule(s)..."
            ):
                pdf = generate_pdf_report(
                    "MULTI-MOLECULE STUDENT ANALYSIS REPORT",
                    name, reg, molecule_records,
                )
            st.success("✅ Combined molecular report generated successfully!")
            st.download_button(
                label="⬇️ Download Combined Molecular Report PDF",
                data=pdf,
                file_name="Cheminformatics_Multi_Molecule_Report.pdf",
                mime="application/pdf",
                key="download_multi_molecule_pdf",
            )
    else:
        st.info("Enter at least one valid SMILES string to generate the report.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💊 " + APP_TITLE + " | Developed by "
    + " & ".join(f"{n} ({r})" for n, r in DEVELOPERS)
    + " | Built with Python, Streamlit and RDKit"
)
