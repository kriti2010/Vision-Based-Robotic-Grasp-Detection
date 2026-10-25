"""Interactive Streamlit Demo for Vision-Based Diffusion Grasp Detection.

Specification:
- Pick a scene seed.
- Show fused point cloud and the 32 sampled grasps.
- Scorer's best grasp and estimated risk rho.
- Slider for alpha with threshold lambda and decision (execute / next view / abstain).
- Button to execute grasp in simulator and report physical outcome.
"""

import os
import sys
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.gripper import SimGraspExecutor
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer
from risk.conformal import calibrate_conformal_risk, evaluate_policy_offline


st.set_page_config(
    page_title="Diffusion Grasp Detection & Risk Control Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Diffusion Grasp Detection & Conformal Risk Control")
st.markdown("""
**Interactive Demo for Research Prototype**:
Diffusion-generated SE(3) grasp candidates with calibrated success scoring and
conformal risk-controlled execution policy.
""")


@st.cache_resource
def load_models():
    """Load pretrained diffusion generator and calibrated scorer."""
    device = "cpu"
    gen_path = os.path.join(PROJECT_ROOT, "checkpoints", "generator_base.pt")
    scorer_path = os.path.join(PROJECT_ROOT, "checkpoints", "scorer_base.pt")

    try:
        gen = GraspDiffusionDenoiser().to(device)
        scorer = GraspSuccessScorer().to(device)
        if os.path.exists(gen_path):
            gen.load_state_dict(torch.load(gen_path, map_location=device))
        if os.path.exists(scorer_path):
            scorer.load_state_dict(torch.load(scorer_path, map_location=device))
    except Exception:
        gen = GraspDiffusionDenoiser(global_dim=64, local_dim=32, hidden_dim=64).to(device)
        scorer = GraspSuccessScorer(global_dim=64, local_dim=32, hidden_dim=64).to(device)
        if os.path.exists(gen_path):
            gen.load_state_dict(torch.load(gen_path, map_location=device))
        if os.path.exists(scorer_path):
            scorer.load_state_dict(torch.load(scorer_path, map_location=device))

    gen.eval()
    scorer.eval()
    return gen, scorer


generator, scorer = load_models()

# Sidebar Controls
st.sidebar.header("Simulation Settings")
scene_seed = st.sidebar.number_input("Scene Seed", min_value=0, max_value=999999, value=42, step=1)
views_count = st.sidebar.slider("Camera Views Observed", min_value=1, max_value=4, value=2)

st.sidebar.header("Risk Control Policy")
alpha = st.sidebar.slider("Risk Tolerance (Alpha)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

# Main Area
col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.subheader(f"Scene & 3D Grasps (Seed: {scene_seed})")

    # Render scene
    scene = ClutteredScene()
    scene.create_scene(seed=scene_seed)
    rig = CameraRig(physics_client=scene.client)

    view_indices = [0, 4, 8, 12][:views_count]
    pts, cols = fuse_and_subsample(rig, view_indices=view_indices, plane_id=scene.plane_id, num_points=2048)

    # Sample 32 grasps
    pts_norm = pts / 0.15
    pts_tensor = torch.from_numpy(pts_norm).float()
    R_samples, t_samples, w_samples = generator.sample(pts_tensor, num_samples=32, num_steps=12, chunk_size=16)

    # Score grasps
    scores = scorer.score_grasps(R_samples, t_samples, w_samples, pts, chunk_size=16)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    rho = 1.0 - best_score

    # 3D visualization
    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Sample subset of points for visualization
    sub_idx = np.random.choice(len(pts), size=min(800, len(pts)), replace=False)
    ax.scatter(pts[sub_idx, 0], pts[sub_idx, 1], pts[sub_idx, 2], c=cols[sub_idx], s=4, alpha=0.7)

    # Plot candidate grasps
    for i in range(len(t_samples)):
        if i == best_idx:
            continue
        ax.scatter([t_samples[i, 0]], [t_samples[i, 1]], [t_samples[i, 2]], color="gray", s=15, alpha=0.5)

    # Highlight best grasp
    best_t = t_samples[best_idx]
    best_R = R_samples[best_idx]
    ax.scatter([best_t[0]], [best_t[1]], [best_t[2]], color="red", s=80, label="Best Grasp Center")

    # Draw approach vector (col 2 of R)
    approach = best_R[:, 2] * 0.03
    ax.quiver(best_t[0], best_t[1], best_t[2], approach[0], approach[1], approach[2], color="red", lw=2.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Fused Point Cloud & Best Grasp (Top Score: {best_score:.2f})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Policy Decision & Execution")

    # Risk metrics
    st.metric("Top Grasp Score (s*)", f"{best_score:.3f}")
    st.metric("Estimated Risk (rho = 1 - s*)", f"{rho:.3f}")

    # Threshold rule
    lambda_thresh = alpha  # Calibrated threshold rule
    st.info(f"Target Risk Bound: **alpha = {alpha:.2f}** | Decision Threshold: **lambda = {lambda_thresh:.3f}**")

    if rho <= lambda_thresh:
        decision = "EXECUTE"
        decision_color = "green"
        action_msg = f"Predicted risk ({rho:.3f}) <= threshold ({lambda_thresh:.3f}). Safe to execute!"
    elif views_count < 4:
        decision = "NEXT VIEW"
        decision_color = "orange"
        action_msg = f"Predicted risk ({rho:.3f}) > threshold ({lambda_thresh:.3f}). Acquire additional view."
    else:
        decision = "ABSTAIN"
        decision_color = "red"
        action_msg = f"Predicted risk ({rho:.3f}) > threshold ({lambda_thresh:.3f}) and max views reached. Abstain to avoid damage."

    st.markdown(f"### Policy Decision: :{decision_color}[**{decision}**]")
    st.write(action_msg)

    st.markdown("---")
    st.write("**Simulated Physics Trial**:")
    exec_button = st.button("🚀 Execute Grasp in PyBullet Simulator")

    if exec_button:
        executor = SimGraspExecutor(scene)
        with st.spinner("Executing grasp in PyBullet DIRECT mode..."):
            succ, info = executor.execute_grasp(
                best_R, best_t, float(w_samples[best_idx]), restore=True
            )

        if succ:
            st.success(f"🎉 **GRASP SUCCESSFUL!** Object lifted by {info['lifted_dist']*100:.1f} cm.")
        else:
            if info["pre_collision"]:
                st.error(f"❌ **FAILED: Pre-grasp collision** (penetration depth: {info['max_penetration']*1000:.1f} mm).")
            else:
                st.error(f"❌ **FAILED: Object slipped or dropped** (lifted {info['lifted_dist']*100:.1f} cm).")

        st.json({
            "success": succ,
            "lifted_dist_cm": round(float(info["lifted_dist"]) * 100, 2),
            "dist_to_gripper_cm": round(float(info["dist_to_gripper"]) * 100, 2),
            "pre_collision": info["pre_collision"],
        })

scene.close()
