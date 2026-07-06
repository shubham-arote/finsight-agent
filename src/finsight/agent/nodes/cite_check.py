"""cite_check — deterministic per-claim verification (the citation rail).

Every figure a claim asserts must appear in the blocks that claim *cites* (not merely
anywhere in the context) or match the verified computation. Failures are surfaced as a
transparent caveat — never silently stripped, never silently trusted. Prose fallback
answers (no claims) get the whole-answer number check instead.
"""

from __future__ import annotations

from ..citations import check_claims
from ..state import Deps, RAGState
from ..verify import verify_numbers


def cite_check(state: RAGState, deps: Deps) -> dict:
    ans = state.get("answer", "")
    if not ans or not state.get("sources"):            # abstained / nothing to check
        return {"unverified": []}
    retrieved = state.get("retrieved", [])
    comp = state.get("computation")
    claims = state.get("claims") or []
    if claims:
        checked, bad = check_claims(claims, retrieved, comp)
    else:
        checked, bad = [], verify_numbers(ans, retrieved, comp)
    if not bad:
        return {"unverified": [], "claims": checked or claims}
    caveat = ("\n\nNote — unverified figure(s) not traceable to the cited evidence: "
              + ", ".join(bad) + " — treat with caution.")
    return {"answer": ans + caveat, "unverified": bad, "claims": checked or claims}
