"""Collision-risk metrics for two surface vessels.

Given two ships described by 2-D position and velocity, this module computes
three standard encounter-risk metrics:

    DCPA  - Distance at Closest Point of Approach
    TCPA  - Time to Closest Point of Approach
    VO    - Velocity Obstacle (collision cone)

The Velocity Obstacle is the set of own-ship velocities that, if held
constant, lead to a collision with the (constant-velocity) target ship within
the combined safety radius.  It is the collision cone in relative-velocity
space translated by the target velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon


class Ship:
    def __init__(self, pos, vel, radius=10.0, name="ship"):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.radius = float(radius)
        self.name = name

    def position_at(self, t):
        """Position after time t under constant velocity (non-mutating)."""
        return self.pos + self.vel * t

    def step(self, t):
        self.pos = self.pos + self.vel * t


class Metrics:
    def __init__(self, ownship, targetship):
        if not isinstance(ownship, Ship) or not isinstance(targetship, Ship):
            raise TypeError("ownship and targetship must be Ship instances")
        self.ownship = ownship
        self.targetship = targetship

        # Relative quantities, target measured w.r.t. own ship.
        self.relpos = targetship.pos - ownship.pos      # line of sight
        self.relvel = targetship.vel - ownship.vel      # closing velocity
        self.safety_radius = ownship.radius + targetship.radius

    # -- Closest point of approach -------------------------------------------
    def TCPA(self):
        """Time to closest point of approach (s). Negative => already past it."""
        speed_sq = float(np.dot(self.relvel, self.relvel))
        if speed_sq < 1e-12:          # parallel, no relative motion
            return 0.0
        return -float(np.dot(self.relpos, self.relvel)) / speed_sq

    def DCPA(self):
        """Distance at closest point of approach (m)."""
        speed_sq = float(np.dot(self.relvel, self.relvel))
        if speed_sq < 1e-12:
            return float(np.linalg.norm(self.relpos))
        t = self.TCPA()
        return float(np.linalg.norm(self.relpos + self.relvel * t))

    def encounter_angle(self):
        """Angle theta between relative velocity and line of sight (rad), [0, pi].

        This single parameter governs the *shape* of the encounter:
            DCPA       = d * sin(theta)
            TCPA       = -(d / v) * cos(theta)
        so the normalized metrics depend only on theta:
            DCPA / d       =  sin(theta)
            TCPA * v / d   = -cos(theta)
        """
        d = float(np.linalg.norm(self.relpos))
        v = float(np.linalg.norm(self.relvel))
        if d < 1e-12 or v < 1e-12:
            return 0.0
        cos_t = float(np.dot(self.relpos, self.relvel)) / (d * v)
        return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    # -- Velocity obstacle ----------------------------------------------------
    def VO(self):
        """Velocity obstacle / collision cone.

        Returns a dict describing the cone and whether the current own-ship
        velocity lies inside it (i.e. is on a collision course):

            apex        : cone apex in velocity space (= target velocity)
            axis        : unit vector along the line of sight (own -> target)
            half_angle  : cone half-angle (rad)
            edges       : the two unit edge directions of the cone
            collision   : bool, True if current own velocity is inside the VO
        """
        dist = float(np.linalg.norm(self.relpos))
        R = self.safety_radius

        if dist <= R:
            # Already overlapping the safety disc: every velocity is unsafe.
            return {
                "apex": self.targetship.vel.copy(),
                "axis": np.zeros(2),
                "half_angle": np.pi,
                "edges": (np.zeros(2), np.zeros(2)),
                "collision": True,
            }

        axis = self.relpos / dist
        half_angle = np.arcsin(R / dist)

        def rotate(v, a):
            c, s = np.cos(a), np.sin(a)
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

        edges = (rotate(axis, half_angle), rotate(axis, -half_angle))

        # Own velocity relative to the cone apex (apex = target velocity).
        w = self.ownship.vel - self.targetship.vel
        wn = np.linalg.norm(w)
        if wn < 1e-12:
            collision = False
        else:
            ang = np.arccos(np.clip(np.dot(w / wn, axis), -1.0, 1.0))
            collision = bool(ang <= half_angle)

        return {
            "apex": self.targetship.vel.copy(),
            "axis": axis,
            "half_angle": half_angle,
            "edges": edges,
            "collision": collision,
        }


def plot_encounter(ownship, targetship, metrics, savepath="risk_metrics.png"):
    """Two-panel research-style figure: encounter geometry + velocity space."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.dpi": 150,
    })

    tcpa = metrics.TCPA()
    dcpa = metrics.DCPA()
    vo = metrics.VO()
    R = metrics.safety_radius

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    c_own, c_tgt = "#1f77b4", "#d62728"
    risk = vo["collision"]

    # ---- Panel 1: encounter geometry (position space) ----------------------
    t_horizon = max(abs(tcpa) * 1.6, 30.0)
    for sh, c in ((ownship, c_own), (targetship, c_tgt)):
        traj = np.array([sh.position_at(t) for t in (0, t_horizon)])
        ax1.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=1.2, alpha=0.6)
        ax1.annotate("", xy=sh.pos + sh.vel * 5, xytext=sh.pos,
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax1.plot(*sh.pos, "o", color=c, ms=9, zorder=5,
                 label=f"{sh.name}  |v|={np.linalg.norm(sh.vel):.1f}")

    # Positions at CPA.
    if tcpa > 0:
        op = ownship.position_at(tcpa)
        tp = targetship.position_at(tcpa)
        ax1.plot(*op, "o", color=c_own, mfc="white", ms=8, zorder=5)
        ax1.plot(*tp, "o", color=c_tgt, mfc="white", ms=8, zorder=5)
        ax1.plot([op[0], tp[0]], [op[1], tp[1]], ":", color="k", lw=1.2)
        mid = (op + tp) / 2
        ax1.annotate(f"DCPA = {dcpa:.1f} m", mid, fontsize=9,
                     ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # Safety disc around target (combined radius).
    ax1.add_patch(Circle(targetship.pos, R, fc=c_tgt, ec=c_tgt, alpha=0.10, lw=1))

    ax1.set_title("Scenario", fontsize=12)
    ax1.set_xlabel("East  [m]")
    ax1.set_ylabel("North  [m]")
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    ax1.set_aspect("equal", adjustable="datalim")

    txt = (f"TCPA = {tcpa:.1f} s\n"
           f"DCPA = {dcpa:.1f} m\n"
           f"$R_{{safe}}$ = {R:.0f} m")
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes, fontsize=9,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    # ---- Panel 2: velocity space (velocity obstacle) -----------------------
    apex = vo["apex"]
    L = 18.0  # cone ray length for drawing
    e1, e2 = vo["edges"]
    cone_color = "#d62728" if risk else "#2ca02c"  # orange-red if unsafe, green if safe
    cone = Polygon([apex, apex + e1 * L, apex + e2 * L],
                   closed=True, fc=cone_color, ec=cone_color, alpha=0.18, lw=1.2)
    ax2.add_patch(cone)
    for e in (e1, e2):
        ax2.plot([apex[0], apex[0] + e[0] * L], [apex[1], apex[1] + e[1] * L],
                 "-", color=cone_color, lw=1.3)

    ax2.annotate("", xy=ownship.vel, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=c_own, lw=2))
    ax2.annotate("", xy=targetship.vel, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=c_tgt, lw=2))
    ax2.plot(*apex, "s", color=c_tgt, ms=7, zorder=5, label="VO apex ($v_T$)")
    ax2.plot(*ownship.vel, "o", color=c_own, ms=9, zorder=6,
             mec=cone_color, mew=2.5, label="$v_O$ (own velocity)")
    ax2.plot(0, 0, "+", color="k", ms=10)

    ax2.set_title("Velocity obstacle", fontsize=12)
    ax2.set_xlabel("$v_{East}$  [m/s]")
    ax2.set_ylabel("$v_{North}$  [m/s]")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Two-Vessel Risk Metrics", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved figure to {savepath}")
    return fig


def plot_metric_landscape(scenarios=None, savepath="risk_metric_landscape.png"):
    """Scenario-independent view of DCPA/TCPA as functions of the encounter
    angle theta (angle between relative velocity and line of sight).

    Because DCPA/d = sin(theta) and TCPA*v/d = -cos(theta), every encounter,
    regardless of range or closing speed, lives on a single universal curve.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.dpi": 150,
    })

    theta = np.linspace(0, np.pi, 400)
    norm_dcpa = np.sin(theta)
    norm_tcpa = -np.cos(theta)
    deg = np.degrees(theta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- Panel 1: normalized metrics vs theta ------------------------------
    ax1.plot(deg, norm_dcpa, color="#1f77b4", lw=2,
             label=r"$\mathrm{DCPA}/d = \sin\theta$")
    ax1.plot(deg, norm_tcpa, color="#d62728", lw=2,
             label=r"$\mathrm{TCPA}\,v/d = -\cos\theta$")
    ax1.axhline(0, color="k", lw=0.8)
    # Past-CPA region (TCPA < 0  <=>  theta < 90 deg): diverging.
    ax1.axvspan(0, 90, color="#2ca02c", alpha=0.07)
    ax1.axvspan(90, 180, color="#d62728", alpha=0.07)
    ax1.text(45, -1.18, "diverging\n(TCPA < 0)", ha="center", va="top",
             fontsize=9, color="#2ca02c")
    ax1.text(135, -1.18, "closing\n(TCPA > 0)", ha="center", va="top",
             fontsize=9, color="#d62728")
    ax1.set_xlim(0, 180)
    ax1.set_xticks(np.arange(0, 181, 30))
    ax1.set_xlabel(r"Encounter angle  $\theta$  [deg]")
    ax1.set_ylabel("Normalized metric")
    ax1.set_title("(a) Normalized DCPA & TCPA vs. encounter angle", fontsize=12)
    ax1.legend(loc="upper center", fontsize=9, framealpha=0.9)

    # ---- Panel 2: universal locus (unit semicircle) ------------------------
    ax2.plot(norm_tcpa, norm_dcpa, color="#555555", lw=2, zorder=2,
             label="universal locus")
    # Shade near-miss / risk band: small normalized DCPA while closing.
    band = norm_dcpa <= 0.3
    ax2.fill_between(norm_tcpa, 0, norm_dcpa, where=band & (norm_tcpa > 0),
                     color="#d62728", alpha=0.15,
                     label=r"risk band ($\mathrm{DCPA}/d \leq 0.3$, closing)")

    if scenarios:
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(scenarios)))
        for (name, m), c in zip(scenarios, cmap):
            th = m.encounter_angle()
            ax2.plot(-np.cos(th), np.sin(th), "o", color=c, ms=10,
                     mec="k", mew=0.8, zorder=5,
                     label=f"{name} ($\\theta$={np.degrees(th):.0f}°)")

    ax2.axvline(0, color="k", lw=0.8, ls=":")
    ax2.set_xlabel(r"Normalized TCPA   $\mathrm{TCPA}\,v/d = -\cos\theta$")
    ax2.set_ylabel(r"Normalized DCPA   $\mathrm{DCPA}/d = \sin\theta$")
    ax2.set_title("(b) Universal DCPA–TCPA locus", fontsize=12)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("DCPA / TCPA as Encounter Metrics", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved figure to {savepath}")
    return fig


if __name__ == "__main__":
    ownship = Ship([0, 0], [5, 0], radius=10, name="Own ship")
    targetship = Ship([0, 200], [5, 0], radius=10, name="Target ship")

    m = Metrics(ownship, targetship)
    print(f"DCPA = {m.DCPA():.2f} m")
    print(f"TCPA = {m.TCPA():.2f} s")
    print(f"VO collision course: {m.VO()['collision']}")

    plot_encounter(ownship, targetship, m)

    # Representative encounters spanning the theta axis. The closing cases share
    # the same relative velocity but differ in lateral miss distance, so they
    # spread along the universal locus instead of piling up near theta = 180.
    def closing(lateral, ahead):
        return Metrics(Ship([0, 0], [0, 3]), Ship([lateral, ahead], [0, -9]))

    scenarios = [
        ("Head-on (collision)", closing(0, 300)),
        ("Crossing",            closing(173, 300)),
        ("Oblique",             closing(520, 300)),
        ("Beam pass",           closing(300, 0)),
        ("Diverging",           Metrics(ownship, targetship)),
    ]
    #plot_metric_landscape(scenarios)
