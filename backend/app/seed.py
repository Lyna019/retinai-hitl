"""
Seed script — populates the disease catalog, mechanisms, lesion vocabulary,
systemic diseases, anatomical regions, active learning config, and demo users.

Run once at first deploy:
    docker compose exec backend python -m app.seed
"""
from .database import SessionLocal, Base, engine
from .models import (
    User, Mechanism, Disease, LesionVocabulary,
    SystemicDisease, AnatomicalRegion, ActiveLearningConfig,
)
from .routers.auth import hash_password


MECHANISMS = [
    ("VASC",   "Vasculaire",     "Atteinte des vaisseaux rétiniens et choroïdiens"),
    ("DEGEN",  "Dégénératif",    "Dégénérescence des photorécepteurs et de l'EPR"),
    ("INFLAM", "Inflammatoire",  "Inflammation rétinienne ou uvéale"),
    ("DYST",   "Dystrophique",   "Dystrophies héréditaires"),
    ("STRUCT", "Structurel",     "Anomalies structurelles, glaucome, décollement"),
    ("TUMOR",  "Tumoral",        "Tumeurs rétiniennes ou choroïdiennes"),
]

DISEASES = [
    # Common chronic (gradable)
    {"code": "DR",       "name_fr": "Rétinopathie diabétique",      "mech": "VASC",   "gradable": True,
     "grades": ["0","1","2","3","4"],
     "grade_labels": {"0":"Absent","1":"Léger","2":"Modéré","3":"Sévère","4":"Proliférant"},
     "desc": "Complication microvasculaire du diabète affectant la rétine."},
    {"code": "GLAUC",    "name_fr": "Glaucome",                      "mech": "STRUCT", "gradable": True,
     "grades": ["DEB","EVO"],
     "grade_labels": {"DEB":"Débutant","EVO":"Évolué"},
     "desc": "Neuropathie optique progressive liée à une pression intraoculaire élevée."},
    {"code": "DMLA",     "name_fr": "DMLA",                          "mech": "DEGEN",  "gradable": True,
     "grades": ["SEC","HUM"],
     "grade_labels": {"SEC":"Sèche","HUM":"Humide"},
     "desc": "Dégénérescence maculaire liée à l'âge, forme sèche ou humide (néovasculaire)."},
    {"code": "HTN_DR",   "name_fr": "Rétinopathie hypertensive",     "mech": "VASC",   "gradable": True,
     "grades": ["1","2","3","4"],
     "grade_labels": {"1":"Grade I","2":"Grade II","3":"Grade III","4":"Grade IV"},
     "desc": "Modifications vasculaires rétiniennes induites par l'hypertension artérielle."},
    # P1 emergencies
    {"code": "OACR",       "name_fr": "OACR — occlusion artère centrale",      "mech": "VASC",   "urgency": "P1",
     "desc": "Occlusion de l'artère centrale de la rétine ; urgence absolue."},
    {"code": "ABACR",      "name_fr": "ABACR — occlusion branche artérielle",  "mech": "VASC",   "urgency": "P1",
     "desc": "Occlusion d'une branche artérielle rétinienne."},
    {"code": "NOIAA",      "name_fr": "NOIAA — neuropathie optique ischémique","mech": "VASC",   "urgency": "P1",
     "desc": "Neuropathie optique ischémique antérieure aiguë non artéritique."},
    # P2 same-day
    {"code": "DR_MAC_OFF", "name_fr": "Décollement de rétine (macula off)",    "mech": "STRUCT", "urgency": "P2",
     "desc": "Décollement rhegmatogène avec atteinte maculaire, urgence chirurgicale."},
    {"code": "GLAUC_AIGU", "name_fr": "Crise de glaucome aigu",                "mech": "STRUCT", "urgency": "P2",
     "desc": "Fermeture aiguë de l'angle avec hypertonie sévère."},
    # Long tail
    {"code": "OVCR",      "name_fr": "OVCR — occlusion veine centrale",      "mech": "VASC",
     "desc": "Occlusion de la veine centrale de la rétine."},
    {"code": "BRVO",      "name_fr": "Occlusion de branche veineuse",        "mech": "VASC",
     "desc": "Occlusion d'une branche veineuse rétinienne."},
    {"code": "CHORIO",    "name_fr": "Choriorétinite",                       "mech": "INFLAM",
     "desc": "Inflammation de la choroïde et de la rétine d'origine infectieuse ou auto-immune."},
    {"code": "UVEITE",    "name_fr": "Uvéite postérieure",                   "mech": "INFLAM",
     "desc": "Inflammation du segment postérieur de l'uvée."},
    {"code": "STARGARDT", "name_fr": "Maladie de Stargardt",                 "mech": "DYST",
     "desc": "Dystrophie maculaire héréditaire autosomique récessive (gène ABCA4)."},
    {"code": "BEST",      "name_fr": "Maladie de Best",                      "mech": "DYST",
     "desc": "Dystrophie maculaire vitelliforme autosomique dominante (gène VMD2)."},
    {"code": "RP",        "name_fr": "Rétinite pigmentaire",                 "mech": "DYST",
     "desc": "Dystrophie rétinienne progressive touchant d'abord les bâtonnets périphériques."},
    {"code": "MELANOME",  "name_fr": "Mélanome choroïdien",                  "mech": "TUMOR",
     "desc": "Tumeur maligne mélanocytaire de la choroïde, la plus fréquente des tumeurs oculaires."},
    {"code": "RETINOBLA", "name_fr": "Rétinoblastome",                       "mech": "TUMOR",
     "desc": "Tumeur maligne rétinienne pédiatrique, mutation du gène RB1."},
    {"code": "MYOPIE_F",  "name_fr": "Myopie forte (fond)",                  "mech": "STRUCT",
     "desc": "Modifications du fond d'œil liées à une myopie > −6 dioptries."},
    {"code": "NORMAL",    "name_fr": "Fond normal",                           "mech": None,
     "desc": "Aucune anomalie rétinienne détectée."},
    # Added for test-set import (ME column in master_dataset_final2)
    {"code": "OED_MAC",  "name_fr": "Œdème maculaire",                      "mech": "VASC",
     "desc": "Accumulation de liquide dans la macula, souvent associée à la rétinopathie diabétique ou à des occlusions veineuses rétiniennes."},
]

LESIONS = [
    ("HEM",  "Hémorragie",     "#FF453A"),
    ("EXS",  "Exsudat dur",    "#FFD60A"),
    ("MA",   "Microanévrisme", "#FF9F0A"),
    ("NV",   "Néovaisseau",    "#BF5AF2"),
    ("DRUS", "Drusen",         "#64D2FF"),
    ("CW",   "Cotton-wool",    "#FFFFFF"),
    ("OED",  "Œdème",          "#30D158"),
]

SYSTEMIC_DISEASES = [
    ("Diabète de type 1",              "Métabolique"),
    ("Diabète de type 2",              "Métabolique"),
    ("Hypertension artérielle",        "Cardiovasculaire"),
    ("Dyslipidémie",                   "Métabolique"),
    ("Obésité",                        "Métabolique"),
    ("Tabagisme",                      "Comportemental"),
    ("Insuffisance rénale chronique",  "Rénal"),
    ("Drépanocytose",                  "Hématologique"),
    ("Lupus érythémateux systémique",  "Auto-immun"),
    ("Polyarthrite rhumatoïde",        "Auto-immun"),
    ("Sarcoïdose",                     "Systémique"),
    ("Maladie de Behçet",              "Systémique"),
    ("HIV / SIDA",                     "Infectieux"),
    ("Tuberculose",                    "Infectieux"),
    ("Toxoplasmose",                   "Infectieux"),
    ("Grossesse",                      "Obstétrical"),
    ("Atteinte thyroïdienne",          "Endocrinien"),
    ("Syndrome d'apnées du sommeil",   "Respiratoire"),
]

ANATOMICAL_REGIONS = [
    "Macula",
    "Fovéa",
    "Disque optique",
    "Arcade vasculaire supérieure",
    "Arcade vasculaire inférieure",
    "Périphérie nasale",
    "Périphérie temporale",
    "Équateur rétinien",
    "Zone péripapillaire",
    "Rétine périphérique (360°)",
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Mechanisms
        for code, name, desc in MECHANISMS:
            if not db.query(Mechanism).filter(Mechanism.code == code).first():
                db.add(Mechanism(code=code, name_fr=name, description=desc))

        # Diseases
        for d in DISEASES:
            existing = db.query(Disease).filter(Disease.code == d["code"]).first()
            if not existing:
                db.add(Disease(
                    code=d["code"],
                    name_fr=d["name_fr"],
                    description=d.get("desc"),
                    mechanism_code=d.get("mech"),
                    is_gradable=d.get("gradable", False),
                    grades_json=d.get("grades"),
                    grade_labels_json=d.get("grade_labels"),
                    urgency_override=d.get("urgency"),
                    is_approved=True,
                ))
            else:
                # Back-fill descriptions and grade labels on existing rows
                if not existing.description and d.get("desc"):
                    existing.description = d["desc"]
                if not existing.grade_labels_json and d.get("grade_labels"):
                    existing.grade_labels_json = d["grade_labels"]

        # Lesions
        for code, name, color in LESIONS:
            if not db.query(LesionVocabulary).filter(LesionVocabulary.code == code).first():
                db.add(LesionVocabulary(code=code, name_fr=name, color_hex=color))

        # Systemic diseases
        for name, category in SYSTEMIC_DISEASES:
            if not db.query(SystemicDisease).filter(SystemicDisease.name_fr == name).first():
                db.add(SystemicDisease(name_fr=name, category=category))

        # Anatomical regions
        for region_name in ANATOMICAL_REGIONS:
            if not db.query(AnatomicalRegion).filter(AnatomicalRegion.name_fr == region_name).first():
                db.add(AnatomicalRegion(name_fr=region_name, is_custom=False))

        # Active learning config (singleton)
        if not db.query(ActiveLearningConfig).filter(ActiveLearningConfig.id == "singleton").first():
            db.add(ActiveLearningConfig())

        # Demo users
        if not db.query(User).filter(User.username == "mekki").first():
            db.add(User(
                username="mekki",
                email="mekki@iah.dz",
                full_name="Dr. Moatez Billah Mekki",
                role="doctor",
                password_hash=hash_password("demo"),
            ))
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(
                username="admin",
                full_name="Administrateur",
                role="admin",
                password_hash=hash_password("admin"),
            ))

        db.commit()
        print("✓ Seed terminé.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
