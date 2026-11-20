"""Common colours and style of the report figures (one figure = one script = one PDF)."""

import matplotlib as mpl

# Fig. 1 left: the parts of the noise the BPC predicts (methods.md §6.2).
NOISE_COLORS = {
    "template": "#1f4e79",  # reproducible from one print to the next
    "print": "#5b9bd5",  # specific to one print, reproducible from one capture to the next
    "capture": "#d6e3f3",  # changes from one capture to the next
    "print_and_capture": "#a7c5e8",  # Scantrust: no repeated scan, the two are not split
}


def setup() -> None:
    """Matplotlib settings shared by all figures: small fonts, fonts embedded in PDFs."""
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
