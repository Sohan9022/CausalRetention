"""
CausalRetention Brand Identity & Vector Emblem Generator
Top-tier Google DeepMind / Meta AI Research Prototype Visual Standard
"""

def get_logo_svg(size: int = 40, show_glow: bool = True) -> str:
    """
    Returns an inline, standalone scalable SVG string representing the Causal Nexus emblem.
    - Represents counterfactual orbits (Y0 control, Y1 treatment), intersecting at the uplift vector (tau).
    - Uses Google DeepMind / Meta AI multi-stop luminous gradient (Indigo -> Violet -> Cyan -> Emerald).
    """
    glow_filter = """
    <filter id="coreGlow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    """ if show_glow else ""

    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: inline-block; vertical-align: middle;">
  <defs>
    <linearGradient id="bgGrad_{size}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#141E33"/>
      <stop offset="100%" stop-color="#090D18"/>
    </linearGradient>
    <linearGradient id="borderGrad_{size}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6366F1" stop-opacity="0.95"/>
      <stop offset="50%" stop-color="#8B5CF6" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#06B6D4" stop-opacity="0.95"/>
    </linearGradient>
    <linearGradient id="upliftGrad_{size}" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6366F1"/>
      <stop offset="35%" stop-color="#8B5CF6"/>
      <stop offset="70%" stop-color="#06B6D4"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
    <linearGradient id="controlGrad_{size}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#6366F1"/>
    </linearGradient>
    <linearGradient id="treatmentGrad_{size}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#06B6D4"/>
    </linearGradient>
    {glow_filter}
  </defs>

  <!-- Precision Squircle Badge -->
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#bgGrad_{size})" stroke="url(#borderGrad_{size})" stroke-width="1.8"/>

  <!-- Subtle Counterfactual Coordinate Ring -->
  <circle cx="32" cy="32" r="21" stroke="rgba(255, 255, 255, 0.08)" stroke-width="1" stroke-dasharray="2 3"/>

  <!-- Y0 Control Counterfactual Orbit Arc -->
  <path d="M16 38 C 16 26, 26 18, 32 18 C 36 18, 41 21, 44 26" stroke="url(#controlGrad_{size})" stroke-width="2.5" stroke-linecap="round"/>

  <!-- Y1 Treatment Counterfactual Orbit Arc -->
  <path d="M48 26 C 48 38, 38 46, 32 46 C 28 46, 23 43, 20 38" stroke="url(#treatmentGrad_{size})" stroke-width="2.5" stroke-linecap="round"/>

  <!-- Central Causal Lift Prism (Delta / Tau Arrow) -->
  <path d="M32 14 L42 36 L32 31 L22 36 Z" fill="url(#upliftGrad_{size})" filter="url(#coreGlow)"/>
  
  <!-- Counterfactual Convergence Apex -->
  <circle cx="32" cy="14" r="3.2" fill="#FFFFFF"/>

  <!-- Dynamic Causal Nodes -->
  <circle cx="18" cy="38" r="2.5" fill="#38BDF8"/>
  <circle cx="46" cy="26" r="2.5" fill="#34D399"/>
  <circle cx="32" cy="46" r="2" fill="#818CF8"/>

  <!-- Core Uplift Spine -->
  <line x1="32" y1="18" x2="32" y2="31" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round" opacity="0.9"/>
</svg>"""
    return svg
