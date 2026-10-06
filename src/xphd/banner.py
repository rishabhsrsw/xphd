"""The banner an xphd command prints when it starts, in the spirit of yambo's.

The lettering was generated once with pyfiglet and is stored here, so the
package has no runtime dependency on it. Like yambo, the face changes from run
to run. Set XPHD_NO_BANNER=1 to switch the banner off (for example in batch
logs); it is never printed for --help or --version.
"""
from __future__ import annotations

import os
import platform
import random
import sys
import time

ART = {
    'big': "__   _______  _     _____\n\\ \\ / /  __ \\| |   |  __ \\\n \\ V /| |__) | |__ | |  | |\n  > < |  ___/| '_ \\| |  | |\n / . \\| |    | | | | |__| |\n/_/ \\_\\_|    |_| |_|_____/",
    'slant': '   _  __ ____  __    ____\n  | |/ // __ \\/ /_  / __ \\\n  |   // /_/ / __ \\/ / / /\n /   |/ ____/ / / / /_/ /\n/_/|_/_/   /_/ /_/_____/',
    'doom': "__   ________ _    ______\n\\ \\ / /| ___ \\ |   |  _  \\\n \\ V / | |_/ / |__ | | | |\n /   \\ |  __/| '_ \\| | | |\n/ /^\\ \\| |   | | | | |/ /\n\\/   \\/\\_|   |_| |_|___/",
}

TAGLINE = "Exciton-Phonon Dynamics"
AUTHORS = ("R. Saraswat", "S. Bhattacharya", "R. Verma", "M. Ansari")
AFFILIATION = "Indian Institute of Information Technology, Allahabad"
PIPELINE = "Quantum ESPRESSO + LetzElPhC + yambo  ->  linewidths, symmetry, optics, transport"


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
              "",
              f"  {PIPELINE}",
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
