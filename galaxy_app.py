"""
Galaxy Morphology Evolution Lab — Streamlit App
================================================
Interactive dashboard for evaluators to explore:
  - Phase 0 : EDA (class distribution, color, structure, redshift)
  - Phase 1 : Baseline model results (ResNet-18, EfficientNet-B0)
  - Phase 2 : Multimodal fusion results + comparison
  - Phase 3 : Centroid Drift Analysis (Phase 3 science)
  - Phase 4 : UMAP / t-SNE galaxy maps
  - Live    : Real-time galaxy classifier (metadata → class)

Run:
    uv run streamlit run galaxy_app.py
"""

# ── std ─────────────────────────────────────────────────────────────────────
import os
import io
import random

# ── third-party ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import torch

# ── project ─────────────────────────────────────────────────────────────────
# Add project dir to path so we can import model files
import sys
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Galaxy Morphology Evolution Lab",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d0d1a; }

.hero-title {
    font-size: 3rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-sub {
    text-align: center; color: #8892b0; font-size: 1.1rem; margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 16px;
    padding: 1.5rem; text-align: center; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 2.8rem; font-weight: 900; color: #64ffda;
    line-height: 1;
}
.metric-label {
    font-size: 0.85rem; color: #8892b0; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 1px;
}
.metric-sub {
    font-size: 0.95rem; color: #ccd6f6; margin-top: 0.3rem;
}
.section-header {
    font-size: 1.5rem; font-weight: 700; color: #ccd6f6;
    border-left: 4px solid #64ffda; padding-left: 0.75rem;
    margin: 1.75rem 0 1rem 0;
}
.finding-card {
    background: #112240; border-left: 4px solid #64ffda;
    border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    color: #ccd6f6;
}
.finding-card b { color: #64ffda; }
.stSelectbox label, .stSlider label { color: #ccd6f6 !important; }
div[data-testid="stSidebar"] { background: #0a0a1a; border-right: 1px solid #0f3460; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["disk", "edge_on", "smooth", "spiral"]
CLASS_COLORS = {"disk": "#45B7D1", "edge_on": "#FFA07A", "smooth": "#FF6B6B", "spiral": "#4ECDC4"}

OUTPUT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "outputs")
CHECKPOINT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoints")
IMG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images_gz2")
MASTER_CSV = os.path.join(os.path.abspath(os.path.dirname(__file__)), "galaxy_master_dataset.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_master():
    if not os.path.exists(MASTER_CSV):
        return None
    df = pd.read_csv(MASTER_CSV)
    df["image_path"] = df["image_path"].apply(
        lambda p: os.path.join(IMG_DIR, os.path.basename(str(p)))
    )
    return df


@st.cache_data(show_spinner=False)
def load_drift_csv(backbone):
    path = os.path.join(OUTPUT_DIR, f"drift_{backbone}", "centroid_drift.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_embeddings_phase4(backbone, method):
    d = os.path.join(OUTPUT_DIR, f"phase4_{backbone}")
    coords_path = os.path.join(d, f"{method}_coords.npy")
    idx_path    = os.path.join(d, f"{method}_sample_idx.npy")
    labels_path = os.path.join(OUTPUT_DIR, f"drift_{backbone}", "embedding_labels.csv")
    if not all(os.path.exists(p) for p in [coords_path, idx_path, labels_path]):
        return None, None
    coords = np.load(coords_path)
    idx    = np.load(idx_path)
    labels = pd.read_csv(labels_path)
    return coords, labels.iloc[idx].reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_model(backbone):
    """Load the best multimodal checkpoint for the given backbone."""
    from multimodal_model import MultimodalFusionNet
    ckpt_path  = os.path.join(CHECKPOINT_DIR, f"multimodal_{backbone}.pth")
    stats_path = os.path.join(CHECKPOINT_DIR, "metadata_stats.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusionNet(num_classes=4, backbone=backbone, freeze_backbone=False)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if ckpt is a dict with 'model_state_dict' or just the state_dict itself
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device).eval()
    stats = torch.load(stats_path, map_location="cpu", weights_only=False) if os.path.exists(stats_path) else None
    return model, stats


def get_image(row_or_path):
    """Load a PIL image from a path or dataframe row."""
    try:
        path = row_or_path if isinstance(row_or_path, str) else row_or_path["image_path"]
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Galaxy Lab")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Home", "EDA", "Model Performance", "Drift Analysis", "Galaxy Map", "Live Classifier"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#8892b0; font-size:0.8rem;'>"
        "Galaxy Zoo 2 (Hart et al. 2016)<br>47,089 galaxies · SDSS DR7"
        "</div>", unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown('<div class="hero-title">Galaxy Morphology Evolution Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">How do galaxies change across cosmic time? We trained a multimodal deep learning model to find out.</div>', unsafe_allow_html=True)

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">47k</div><div class="metric-label">Galaxies</div><div class="metric-sub">SDSS DR7 cutouts</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">78.5%</div><div class="metric-label">Best Accuracy</div><div class="metric-sub">Multimodal ResNet-18</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">4.07</div><div class="metric-label">Max Drift</div><div class="metric-sub">Disk · low→mid z</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">256-d</div><div class="metric-label">Embedding Space</div><div class="metric-sub">Fused image + metadata</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline overview
    st.markdown('<div class="section-header">Project Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    phases = [
        ("Phase 1", "Baseline Classifier", "ResNet-18 & EfficientNet-B0 image-only models. 76–79% accuracy."),
        ("Phase 2", "Multimodal Fusion", "Fused image embeddings with spectroscopic metadata (redshift, color, structure)."),
        ("Phase 3", "Centroid Drift", "Extracted 256-d embeddings for all 47k galaxies. Measured class centroid shift across redshift bins."),
        ("Phase 4", "UMAP / t-SNE", "2D visualisation of the embedding space. Clusters reveal learned morphological structure."),
    ]
    for col, (phase, title, desc) in zip([p1, p2, p3, p4], phases):
        with col:
            st.markdown(f"**{phase}**")
            st.markdown(f"##### {title}")
            st.markdown(f"<span style='color:#8892b0;font-size:0.9rem;'>{desc}</span>", unsafe_allow_html=True)

    st.markdown("---")
    # Key scientific findings
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    findings = [
        ("Disk galaxies drift the most", "Euclidean centroid shift of <b>4.07</b> from low-z → mid-z (ResNet-18 model). Likely reflects SDSS resolution degradation at higher redshifts."),
        ("Spiral galaxies are the most stable", "Mid → high-z drift of only <b>0.99</b> — spiral arm features remain detectable the longest."),
        ("Smooth galaxies extend furthest", "Only smooth galaxies have statistical validity beyond z = 0.14 (2,210 galaxies in extended bin), confirming the SDSS resolution effect."),
        ("Multimodal > Image-only", "Adding redshift, color, and structural metadata improved accuracy by ~2% and F1 by ~4% over the image-only baseline."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="finding-card"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "EDA":
    st.markdown("# Exploratory Data Analysis")
    st.markdown("An overview of the Galaxy Zoo 2 dataset before any modelling was done.")

    df = load_master()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Class Distribution", "Redshift", "Colors", "Structure", "Sample Images"])

    with tab1:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        eda_img = os.path.join(OUTPUT_DIR, "eda", "class_distribution.png")
        if os.path.exists(eda_img):
            st.image(eda_img, use_container_width=True)

        if df is not None:
            counts = df["morph_class"].value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
            fig = px.bar(
                counts, x="Class", y="Count", color="Class",
                color_discrete_map=CLASS_COLORS,
                text=counts["Pct"].astype(str) + "%",
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, title="Galaxy Class Counts")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Redshift Distribution</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, "eda", "redshift_per_class.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

        if df is not None:
            fig = px.histogram(
                df, x="redshift", color="morph_class",
                color_discrete_map=CLASS_COLORS,
                nbins=80, barmode="overlay", opacity=0.7,
                template="plotly_dark",
                title="Redshift Distribution per Class",
                labels={"redshift": "Redshift (z)", "morph_class": "Class"},
            )
            fig.add_vline(x=0.06, line_dash="dash", line_color="white", annotation_text="low/mid boundary")
            fig.add_vline(x=0.10, line_dash="dash", line_color="yellow", annotation_text="mid/high boundary")
            fig.add_vline(x=0.14, line_dash="dash", line_color="red", annotation_text="ext boundary")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Color–Redshift Correlations</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, "eda", "color_redshift.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

        if df is not None:
            fig = px.scatter(
                df.sample(5000, random_state=42),
                x="redshift", y="color_u_r",
                color="morph_class", color_discrete_map=CLASS_COLORS,
                opacity=0.4, template="plotly_dark",
                title="u−r Color vs Redshift",
                labels={"redshift": "Redshift (z)", "color_u_r": "u−r (extinction corrected)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-header">Structural Features</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, "eda", "structural_features.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

        if df is not None:
            fig = px.box(
                df, x="morph_class", y="fracdev_r",
                color="morph_class", color_discrete_map=CLASS_COLORS,
                template="plotly_dark",
                title="FRACDEV_R by Class (0=disk, 1=elliptical profile)",
                labels={"morph_class": "Class", "fracdev_r": "FRACDEV_R"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-header">Sample Galaxy Images</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, "eda", "sample_images.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

        if df is not None:
            st.markdown("#### Browse Random Galaxies by Class")
            sel_cls = st.selectbox("Filter by class", ["all"] + CLASS_NAMES)
            subset = df if sel_cls == "all" else df[df["morph_class"] == sel_cls]
            sample = subset.sample(min(12, len(subset)), random_state=random.randint(0, 999))
            cols = st.columns(6)
            for i, (_, row) in enumerate(sample.iterrows()):
                img_pil = get_image(row)
                if img_pil:
                    with cols[i % 6]:
                        st.image(img_pil, caption=f"{row['morph_class']} · z={row['redshift']:.3f}", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("# Model Performance")

    # Summary table
    st.markdown('<div class="section-header">All Models at a Glance</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([
        {"Model": "ResNet-18 (image only)",       "Type": "Baseline",   "Test Acc": "76.7%", "Test F1": "0.66"},
        {"Model": "EfficientNet-B0 (image only)", "Type": "Baseline",   "Test Acc": "78.8%", "Test F1": "0.68"},
        {"Model": "MM EfficientNet-B0",           "Type": "Multimodal", "Test Acc": "75.3%", "Test F1": "0.67"},
        {"Model": "MM ResNet-18 (Best)",              "Type": "Multimodal", "Test Acc": "78.5%", "Test F1": "0.71"},
    ])
    st.dataframe(summary.style.highlight_max(axis=0, subset=["Test Acc", "Test F1"], color="#064e3b"),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    model_choice = st.selectbox(
        "Select model to inspect",
        ["ResNet-18 (Baseline)", "EfficientNet-B0 (Baseline)", "Multimodal EfficientNet-B0", "Multimodal ResNet-18"],
    )

    model_map = {
        "ResNet-18 (Baseline)":         ("baseline",                  "training_curves.png", "confusion_matrix.png", None),
        "EfficientNet-B0 (Baseline)":   ("efficientnet",              "training_curves.png", "confusion_matrix.png", None),
        "Multimodal EfficientNet-B0":   ("multimodal_efficientnet_b0","training_curves.png", "confusion_matrix.png", "baseline_comparison.png"),
        "Multimodal ResNet-18":         ("multimodal_resnet18",       "training_curves.png", "confusion_matrix.png", "baseline_comparison.png"),
    }

    folder, curves_f, cm_f, comp_f = model_map[model_choice]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        p = os.path.join(OUTPUT_DIR, folder, curves_f)
        if os.path.exists(p):
            st.image(p, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        p = os.path.join(OUTPUT_DIR, folder, cm_f)
        if os.path.exists(p):
            st.image(p, use_container_width=True)

    if comp_f:
        st.markdown('<div class="section-header">Baseline vs Multimodal Comparison</div>', unsafe_allow_html=True)
        p = os.path.join(OUTPUT_DIR, folder, comp_f)
        if os.path.exists(p):
            st.image(p, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DRIFT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Drift Analysis":
    st.markdown("# Centroid Drift Analysis")
    st.markdown(
        "Each galaxy class is placed into a **256-dimensional embedding space** by the fusion model. "
        "We group galaxies by redshift bin and measure how the class *centroid* (average position) moves "
        "across cosmic time — a proxy for morphological evolution."
    )

    backbone = st.selectbox("Select model backbone", ["resnet18", "efficientnet_b0"],
                            format_func=lambda x: "ResNet-18 (Best)" if x == "resnet18" else "EfficientNet-B0")

    col_img, col_pop = st.columns(2)
    with col_img:
        st.markdown('<div class="section-header">Drift Curves</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, f"drift_{backbone}", "drift_curves.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
    with col_pop:
        st.markdown('<div class="section-header">Bin Population</div>', unsafe_allow_html=True)
        img = os.path.join(OUTPUT_DIR, f"drift_{backbone}", "bin_population.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

    st.markdown('<div class="section-header">Raw Drift Numbers</div>', unsafe_allow_html=True)
    df_drift = load_drift_csv(backbone)
    if df_drift is not None:
        # Normalise column names (actual CSV uses: class, bin_from, bin_to, euclidean_drift, cosine_drift)
        if "transition" not in df_drift.columns:
            df_drift["transition"]  = df_drift["bin_from"] + " \u2192 " + df_drift["bin_to"]
        if "morph_class" not in df_drift.columns:
            df_drift["morph_class"] = df_drift["class"]
        if "euclidean" not in df_drift.columns:
            df_drift["euclidean"]   = df_drift["euclidean_drift"]

        st.dataframe(df_drift, use_container_width=True, hide_index=True)

        # Interactive drift bar chart
        fig = px.bar(
            df_drift, x="transition", y="euclidean",
            color="morph_class", color_discrete_map=CLASS_COLORS,
            barmode="group", template="plotly_dark",
            title=f"Euclidean Centroid Drift ({backbone})",
            labels={"euclidean": "Euclidean Distance", "transition": "Redshift Transition", "morph_class": "Class"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Side-by-side comparison of both backbones
    st.markdown('<div class="section-header">ResNet-18 vs EfficientNet-B0 Comparison</div>', unsafe_allow_html=True)
    df_r = load_drift_csv("resnet18")
    df_e = load_drift_csv("efficientnet_b0")
    if df_r is not None and df_e is not None:
        for _df in [df_r, df_e]:
            if "transition" not in _df.columns:
                _df["transition"]  = _df["bin_from"] + " \u2192 " + _df["bin_to"]
            if "morph_class" not in _df.columns:
                _df["morph_class"] = _df["class"]
            if "euclidean" not in _df.columns:
                _df["euclidean"]   = _df["euclidean_drift"]
        df_r["backbone"] = "ResNet-18"
        df_e["backbone"] = "EfficientNet-B0"
        df_all = pd.concat([df_r, df_e])
        fig2 = px.bar(
            df_all, x="transition", y="euclidean",
            color="backbone", barmode="group",
            facet_col="morph_class", template="plotly_dark",
            title="Backbone Comparison: Euclidean Drift per Class",
            labels={"euclidean": "Euclidean Distance", "transition": "Transition"},
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GALAXY MAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Galaxy Map":
    st.markdown("# Galaxy Embedding Map")
    st.markdown(
        "Each dot is a galaxy, projected from **256 dimensions → 2D** using UMAP or t-SNE. "
        "The clusters reveal the morphological structure the model has learned. "
        "**Hover over any dot** to see its class and redshift. **Click a dot** to view the galaxy image below."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        backbone = st.selectbox("Backbone", ["resnet18", "efficientnet_b0"],
                                format_func=lambda x: "ResNet-18 (Best)" if x == "resnet18" else "EfficientNet-B0")
        method   = st.selectbox("Method", ["umap", "tsne"], format_func=str.upper)
        n_show   = st.slider("Points to show", 1000, 20000, 8000, 1000)

    coords, labels = load_embeddings_phase4(backbone, method)

    with col2:
        if coords is None:
            st.warning(f"No Phase 4 data found for {backbone}/{method}. Run `phase4_visualize.py --backbone {backbone}` first.")
        else:
            # Subsample if needed
            if len(coords) > n_show:
                idx = np.random.choice(len(coords), n_show, replace=False)
                idx.sort()
                coords_s  = coords[idx]
                labels_s  = labels.iloc[idx].reset_index(drop=True)
            else:
                coords_s, labels_s = coords, labels

            fig = go.Figure()
            for cls in CLASS_NAMES:
                mask = labels_s["morph_class"] == cls
                hover = [
                    f"<b>{cls.capitalize()}</b><br>z = {z:.4f}"
                    for z in labels_s.loc[mask, "redshift"]
                ]
                fig.add_trace(go.Scatter(
                    x=coords_s[mask, 0], y=coords_s[mask, 1],
                    mode="markers", name=cls.capitalize(),
                    marker=dict(color=CLASS_COLORS[cls], size=4, opacity=0.7),
                    hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                    customdata=labels_s.loc[mask].index.tolist(),
                ))

            fig.update_layout(
                title=f"Galaxy Morphology — {method.upper()} ({backbone})",
                paper_bgcolor="#0d0d1a", plot_bgcolor="#111122",
                font=dict(color="white"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1),
                height=600,
                hoverlabel=dict(bgcolor="#0f3460", font=dict(color="white", size=13)),
                clickmode="event+select",
            )
            selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="galaxy_map")

            # Image viewer on click/select
            st.markdown('<div class="section-header">Selected Galaxy Viewer</div>', unsafe_allow_html=True)
            if selected and selected.get("selection") and selected["selection"].get("points"):
                pts = selected["selection"]["points"]
                img_cols = st.columns(min(len(pts), 5))
                df_master = load_master()
                if df_master is not None:
                    # ensure dr7objid is string in master for stable matching
                    df_master["dr7objid_str"] = df_master["dr7objid"].astype(str)
                    
                for i, pt in enumerate(pts[:5]):
                    # customdata contains the original index from labels_s
                    # but plotly returns it as a list in pt["customdata"]
                    if "customdata" in pt and len(pt["customdata"]) > 0:
                        idx = pt["customdata"][0]
                        if idx < len(labels_s) and df_master is not None:
                            objid = str(labels_s.iloc[idx].get("dr7objid", ""))
                            subset = df_master[df_master["dr7objid_str"] == objid]
                            if not subset.empty:
                                row = subset.iloc[0]
                                img_pil = get_image(row)
                                if img_pil:
                                    with img_cols[i]:
                                        st.image(img_pil, caption=f"{row['morph_class']} · z={row['redshift']:.4f}", use_container_width=True)
            else:
                st.info("Click on any dot in the map above to view the corresponding galaxy image here.")

    # Static maps for comparison
    st.markdown("---")
    st.markdown('<div class="section-header">Static Comparison (Both Backbones)</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    for col, bb in zip([sc1, sc2], ["resnet18", "efficientnet_b0"]):
        p = os.path.join(OUTPUT_DIR, f"phase4_{bb}", f"{method}_static.png")
        if os.path.exists(p):
            with col:
                st.image(p, caption=f"{method.upper()} — {bb}", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Classifier":
    st.markdown("# Live Galaxy Classifier")
    st.markdown("Upload a galaxy image and enter its SDSS metadata to get an instant morphology prediction from the multimodal model.")

    backbone = st.selectbox("Model backbone", ["resnet18", "efficientnet_b0"],
                            format_func=lambda x: "ResNet-18 (Best)" if x == "resnet18" else "EfficientNet-B0")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("#### Galaxy Image")
        upload_mode = st.radio("Image source", ["Upload image", "Pick random from dataset"])

        img_pil = None
        galaxy_row = None

        if upload_mode == "Upload image":
            uploaded = st.file_uploader("Upload a JPEG galaxy cutout", type=["jpg", "jpeg", "png"])
            if uploaded:
                img_pil = Image.open(uploaded).convert("RGB")
                st.image(img_pil, caption="Uploaded galaxy", use_container_width=True)
        else:
            df = load_master()
            if df is not None:
                sel_cls = st.selectbox("Pick class", ["random"] + CLASS_NAMES)
                if st.button("Pick Random Galaxy"):
                    pool = df if sel_cls == "random" else df[df["morph_class"] == sel_cls]
                    galaxy_row = pool.sample(1).iloc[0]
                    st.session_state["galaxy_row"] = galaxy_row.to_dict()
                if "galaxy_row" in st.session_state:
                    galaxy_row = pd.Series(st.session_state["galaxy_row"])
                    img_pil = get_image(galaxy_row)
                    if img_pil:
                        st.image(img_pil, caption=f"True class: {galaxy_row['morph_class']} · z={galaxy_row['redshift']:.4f}", use_container_width=True)

        st.markdown("#### SDSS Metadata")
        if galaxy_row is not None:
            default_z     = float(galaxy_row.get("redshift", 0.05))
            default_ur    = float(galaxy_row.get("color_u_r", 1.5))
            default_gi    = float(galaxy_row.get("color_g_i", 0.5))
            default_fd    = float(galaxy_row.get("fracdev_r", 0.5))
            default_r50   = float(galaxy_row.get("petror50_r_kpc", 5.0))
            default_mu    = float(galaxy_row.get("mu50_r", 22.0))
        else:
            default_z, default_ur, default_gi = 0.05, 1.5, 0.5
            default_fd, default_r50, default_mu = 0.5, 5.0, 22.0

        redshift   = st.slider("Redshift (z)",        0.01, 0.25, default_z,   0.001, format="%.4f")
        color_ur   = st.slider("Color u−r",           -1.0, 5.0,  default_ur,  0.01)
        color_gi   = st.slider("Color g−i",           -0.5, 3.0,  default_gi,  0.01)
        fracdev    = st.slider("FRACDEV_R",            0.0,  1.0,  default_fd,  0.01)
        petror50   = st.slider("PETROR50_R (kpc)",     0.5,  30.0, default_r50, 0.1)
        mu50       = st.slider("MU50_R (surface brt)", 17.0, 28.0, default_mu,  0.1)

    with col_out:
        st.markdown("#### Prediction")

        if st.button("Classify This Galaxy", type="primary", use_container_width=True):
            if img_pil is None:
                st.warning("Please provide a galaxy image first.")
            else:
                with st.spinner("Running model inference..."):
                    try:
                        model, stats = load_model(backbone)
                    except Exception as load_err:
                        st.error(f"Model load failed: {load_err}")
                        model, stats = None, None
                    if model is None:
                        st.error(f"No checkpoint found for {backbone}. Train the model first.")
                    else:
                        try:
                            from torchvision import transforms
                            device = next(model.parameters()).device

                            # Preprocess image
                            tfm = transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
                            img_t = tfm(img_pil).unsqueeze(0).to(device)

                            # Preprocess metadata
                            meta_raw = torch.tensor([redshift, color_ur, color_gi, fracdev, petror50, mu50], dtype=torch.float32)
                            if stats is not None:
                                meta_raw = (meta_raw - stats["mean"]) / (stats["std"] + 1e-8)
                            meta_t = meta_raw.unsqueeze(0).to(device)

                            with torch.no_grad():
                                logits = model(img_t, meta_t)
                                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

                            pred_idx  = int(probs.argmax())
                            pred_cls  = CLASS_NAMES[pred_idx]
                            pred_conf = probs[pred_idx]

                            # Show prediction badge
                            color = CLASS_COLORS[pred_cls]
                            st.markdown(
                                f"<div style='background:{color}22; border: 2px solid {color}; border-radius:16px; "
                                f"padding:1.5rem; text-align:center; margin-bottom:1rem;'>"
                                f"<div style='font-size:3rem;'>{pred_cls.upper()[0]}</div>"
                                f"<div style='font-size:1.8rem; font-weight:900; color:{color};'>{pred_cls.upper()}</div>"
                                f"<div style='color:#ccd6f6; font-size:1rem;'>Confidence: {pred_conf*100:.1f}%</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # Confidence bar chart
                            fig = px.bar(
                                x=CLASS_NAMES, y=[float(p) for p in probs],
                                color=CLASS_NAMES, color_discrete_map=CLASS_COLORS,
                                template="plotly_dark",
                                labels={"x": "Class", "y": "Probability"},
                                title="Class Probability Distribution",
                            )
                            fig.update_layout(showlegend=False, yaxis_range=[0, 1],
                                              yaxis_tickformat=".0%")
                            st.plotly_chart(fig, use_container_width=True)

                            # Ground truth comparison
                            if galaxy_row is not None:
                                true_cls = galaxy_row.get("morph_class", "unknown")
                                if true_cls == pred_cls:
                                    st.success(f"Correct! True class was **{true_cls}**.")
                                else:
                                    st.error(f"Incorrect. True class was **{true_cls}**.")
                        except Exception as e:
                            st.error(f"Inference error: {e}")

        st.markdown("---")
        st.markdown("#### What the Features Mean")
        with st.expander("Feature explanations"):
            st.markdown("""
| Feature | Description |
|---|---|
| **Redshift (z)** | Cosmological redshift — proxy for distance/lookback time |
| **Color u−r** | Dust-corrected UV to red color index. High = red/elliptical |
| **Color g−i** | Green to infrared color. Separates star-forming vs passive |
| **FRACDEV_R** | De Vaucouleurs profile fraction. 0 = pure disk, 1 = pure elliptical |
| **PETROR50_R (kpc)** | Physical half-light radius in kiloparsecs |
| **MU50_R** | Surface brightness at half-light radius (mag/arcsec²) |
            """)
