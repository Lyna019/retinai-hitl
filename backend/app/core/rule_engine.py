"""
Rule engine: derives mechanism codes and urgency priority from a set of
disease labels. Mirrors the frontend logic so predictions are identical.

Urgency priorities (minimum wins — i.e. P1 beats P3):
  P1 — Urgence vitale          : OACR / ABACR / NOIAA
  P2 — Chirurgie du jour       : Décollement macula-off / Crise de glaucome aigu
  P3 — Suivi urgent            : DR grade 4, DMLA humide, Glaucome évolutive, HTN-DR ≥3
  P4 — Routine                 : everything else
"""
from dataclasses import dataclass

PRIORITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

PRIORITY_LABEL = {
    "P1": "Urgence vitale",
    "P2": "Chirurgie du jour",
    "P3": "Suivi urgent",
    "P4": "Routine",
}

# Minimal static catalog — in production pulled from DB.
DISEASE_META = {
    "DR": {"mechanism": "VASC", "gradable": True},
    "GLAUC": {"mechanism": "STRUCT", "gradable": True},
    "DMLA": {"mechanism": "DEGEN", "gradable": True},
    "HTN_DR": {"mechanism": "VASC", "gradable": True},
    "OACR": {"mechanism": "VASC", "gradable": False, "urgency_override": "P1"},
    "ABACR": {"mechanism": "VASC", "gradable": False, "urgency_override": "P1"},
    "NOIAA": {"mechanism": "VASC", "gradable": False, "urgency_override": "P1"},
    "DR_MAC_OFF": {"mechanism": "STRUCT", "gradable": False, "urgency_override": "P2"},
    "GLAUC_AIGU": {"mechanism": "STRUCT", "gradable": False, "urgency_override": "P2"},
}


@dataclass
class UrgencyResult:
    level: str
    rule: str
    label: str


def compute_mechanisms(labels: list[dict]) -> list[str]:
    out: set[str] = set()
    for label in labels:
        m = DISEASE_META.get(label["disease_code"], {}).get("mechanism")
        if m:
            out.add(m)
    return sorted(out)


def compute_urgency(labels: list[dict]) -> UrgencyResult | None:
    if not labels:
        return None

    best = UrgencyResult(level="P4", rule="Routine", label=PRIORITY_LABEL["P4"])

    for label in labels:
        meta = DISEASE_META.get(label["disease_code"], {})
        override = meta.get("urgency_override")
        code = label["disease_code"]
        grade = label.get("grade")

        if override and PRIORITY_ORDER[override] < PRIORITY_ORDER[best.level]:
            best = UrgencyResult(level=override, rule=code, label=PRIORITY_LABEL[override])
            continue

        if code == "DR" and grade == "4" and PRIORITY_ORDER["P3"] < PRIORITY_ORDER[best.level]:
            best = UrgencyResult(level="P3", rule="DR Proliférative", label=PRIORITY_LABEL["P3"])
        if code == "DMLA" and grade == "HUM" and PRIORITY_ORDER["P3"] < PRIORITY_ORDER[best.level]:
            best = UrgencyResult(level="P3", rule="DMLA Humide", label=PRIORITY_LABEL["P3"])
        if code == "GLAUC" and grade == "EVO" and PRIORITY_ORDER["P3"] < PRIORITY_ORDER[best.level]:
            best = UrgencyResult(level="P3", rule="Glaucome évolutive", label=PRIORITY_LABEL["P3"])
        if code == "HTN_DR" and grade in {"3", "4"} and PRIORITY_ORDER["P3"] < PRIORITY_ORDER[best.level]:
            best = UrgencyResult(level="P3", rule=f"HTN-DR stade {grade}", label=PRIORITY_LABEL["P3"])

    return best
