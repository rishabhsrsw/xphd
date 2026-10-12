"""Internal module of xphd, used by the xphd commands; not run directly."""
from __future__ import annotations

import os
import platform
import random
import sys
import textwrap
import time

ART = {
    'big': "__   _______  _     _____\n\\ \\ / /  __ \\| |   |  __ \\\n \\ V /| |__) | |__ | |  | |\n  > < |  ___/| '_ \\| |  | |\n / . \\| |    | | | | |__| |\n/_/ \\_\\_|    |_| |_|_____/",
    'slant': '   _  __ ____  __    ____\n  | |/ // __ \\/ /_  / __ \\\n  |   // /_/ / __ \\/ / / /\n /   |/ ____/ / / / /_/ /\n/_/|_/_/   /_/ /_/_____/',
    'doom': "__   ________ _    ______\n\\ \\ / /| ___ \\ |   |  _  \\\n \\ V / | |_/ / |__ | | | |\n /   \\ |  __/| '_ \\| | | |\n/ /^\\ \\| |   | | | | |/ /\n\\/   \\/\\_|   |_| |_|___/",
}

TAGLINE = "Exciton-Phonon Dynamics"
AUTHORS = ("Rishabh Saraswat", "Mantasha Ansari", "Rekha Verma",
           "Sitangshu Bhattacharya")
AFFILIATION = "Indian Institute of Information Technology, Allahabad"
PIPELINE = "Quantum ESPRESSO + LetzElPhC + yambo  ->  linewidths, symmetry, optics, transport"

# ==========================================================================
# CITATION -- printed in the banner of every xphd command.
#   PAPER = ("R. Saraswat, M. Ansari, R. Verma and S. Bhattacharya, "
#            "Real-time relaxation dynamics of dark excitons in "
#            "mirror-symmetric monolayers, arXiv:2610.xxxxx (2026)")
PAPER = ""
SOFTWARE = "xphd, Zenodo, https://zenodo.org/records/23232733"

WIDTH = 78                      # citation lines are wrapped to this width


def _citation_lines():
    refs = [r for r in (PAPER, SOFTWARE) if r and r.strip()]
    if not refs:
        return []
    out = ["  If you use xphd in published work, please cite:"]
    for r in refs:
        out += textwrap.wrap(r.strip(), WIDTH - 6,
                             initial_indent="    - ", subsequent_indent="      ")
    return out


def banner_text(version: str, argv=None, face: str | None = None) -> str:
    """The banner as a string; `face` picks the lettering (default: random)."""
    art = ART[face] if face in ART else random.choice(list(ART.values()))
    try:
        import numpy
        npv = numpy.__version__
    except Exception:                       # pragma: no cover
        npv = "?"
    lines = [""]
    lines += ["  " + l for l in art.splitlines()]
    lines += ["",
              f"  Version {version}  --  {TAGLINE}",
              f"  {', '.join(AUTHORS[:-1])} and {AUTHORS[-1]}",
              f"  {AFFILIATION}",
              ""]
    cite = _citation_lines()
    if cite:
        lines += cite + [""]
    lines += [f"  {PIPELINE}",
              f"  python {platform.python_version()} | numpy {npv} | "
              f"{time.strftime('%Y-%m-%d %H:%M')}"]
    if argv:
        lines.append("  $ xphd " + " ".join(str(a) for a in argv))
    lines.append("")
    return "\n".join(lines)


def banner(version: str, argv=None, stream=None) -> None:
    """Print the banner unless XPHD_NO_BANNER is set."""
    if os.environ.get("XPHD_NO_BANNER"):
        return
    print(banner_text(version, argv), file=stream or sys.stdout, flush=True)
