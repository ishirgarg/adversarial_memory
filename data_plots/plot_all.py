"""Run every plotting script in this directory."""
from __future__ import annotations

import plot_success_vs_k
import plot_errors_vs_k
import plot_coexisting_by_n_facts


if __name__ == "__main__":
    plot_success_vs_k.main()
    plot_errors_vs_k.main()
    plot_coexisting_by_n_facts.main()
