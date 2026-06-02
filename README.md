# RetinAI · HITL Platform

Plateforme d'annotation Human-in-the-Loop pour fonds d'œil — diagnostic assisté
par IA, validation Grad-CAM, file d'attente active learning.

> **État actuel** · Frontend complet · Backend skeleton · Modèle stubbé · Docker prêt
> **Cible** · Déploiement local → serveur LAN du CHU
> **Échéance thèse** · juin 2026

---

## Architecture

```
                        ┌─────────────────────────┐
                        │   Frontend  (React)     │  port 80
                        │   Tailwind · Framer     │
                        └────────────┬────────────┘
                                     │ /api/*
                        ┌────────────▼────────────┐
                        │   Backend  (FastAPI)    │  port 8000
                        │   Auth · Annotations    │
                        │   Rule engine · Whisper │
                        └──┬─────────────────┬────┘
                           │                 │
              ┌────────────▼────┐  ┌─────────▼──────────┐
              │ PostgreSQL      │  │ Model microservice │  port 9000
              │ patients,       │  │ FastAPI · PyTorch  │
              │ images,         │  │ predict + gradcam  │
              │ annotations…    │  │ + uncertainty      │
              └─────────────────┘  └────────────────────┘

   [Clinical DB]  ── sync ──►  Backend  (stub now, real connection at the hospital)
```

### Pourquoi ces choix ?

| Choix | Raison |
|---|---|
| **React + Tailwind** | Composants, responsive natif (Dr. Mekki sur tablette/téléphone) |
| **FastAPI** | Async, intégration Pytorch native, OpenAPI auto |
| **PostgreSQL** | Demandé · robuste pour relations + JSON |
| **Modèle dans un microservice** | Hot-swap du `.pth` sans redéployer la plateforme |
| **Docker Compose** | Une commande pour tout lancer · même image dev/CHU |
| **Whisper FR + fallback browser** | Dictée live même sans GPU sur le serveur |

---

## Démarrage rapide

### 1. Cloner et configurer

```bash
git clone <repo>
cd retinai-hitl
cp .env.example .env
# Éditer .env si besoin (mot de passe DB, secret JWT)
```

### 2. Lancer la stack complète

```bash
docker compose up --build
```

Premier démarrage ≈ 2 min (build des images).

### 3. Initialiser le catalogue + utilisateur démo

```bash
docker compose exec backend python -m app.seed
```

### 4. Ouvrir l'application

| Service | URL | Identifiants |
|---|---|---|
| Frontend | http://localhost | `mekki` / `demo` |
| Backend API docs | http://localhost:8000/api/docs | — |
| Modèle (santé) | http://localhost:9000/health | — |

---

## Modules du frontend

| Page | Route | Description |
|---|---|---|
| Login | `/login` | Authentification doctorale |
| Annotation | `/annotate` | Écran principal — file + viewer + panneau |
| File | `/queue` | Vue grille de toutes les images avec filtres |
| Admin | `/admin` | Tableau de bord — productivité, dataset, urgence, HITL, médecins |

### Workflow d'annotation
1. Image actuelle ← file d'attente (triée par incertitude)
2. Modèle propose top-3 prédictions avec confiance
3. Médecin coche pathologies (multi-sélection)
4. Si pathologie gradable → grade pickers apparaissent inline
5. Mécanisme(s) + urgence calculés automatiquement (rule engine)
6. Médecin peint régions d'intérêt sur la grille dynamique (8→16→32→64 cellules selon zoom)
7. Médecin valide la heatmap Grad-CAM (✓ / ◐ / ✗)
8. Notes additionnelles (texte ou dictée vocale FR live)
9. Soumettre → image marquée terminée, prochaine chargée

### Règles d'urgence (auto)

| Priorité | Pathologies | Délai |
|---|---|---|
| **P1** Urgence vitale | OACR / ABACR / NOIAA | Immédiat |
| **P2** Chirurgie du jour | Décollement macula-off / Glaucome aigu | < 24h |
| **P3** Suivi urgent | DR-4 / DMLA humide / Glaucome évolutive / HTN-DR ≥ stade 3 | Jours |
| **P4** Routine | Tout le reste | Suivi régulier |

→ Sur diagnostic multiple, **la priorité minimale gagne** (P1 > P3).

---

## Brancher un vrai modèle

Quand le checkpoint PyTorch est prêt :

```bash
# 1. Copier le .pth
cp retinai-v1.0.pth model-service/checkpoints/retinai.pth

# 2. Activer torch
# Décommenter torch + torchvision dans model-service/requirements.txt

# 3. Implémenter _run_inference() dans model-service/app/main.py
#    (le contrat d'entrée/sortie est déjà défini)

# 4. Mettre à jour la version
echo "MODEL_VERSION=retinai-v1.0" >> .env

# 5. Rebuild juste le service modèle
docker compose up --build model-service
```

Le frontend, le backend, et la base ne sont pas redémarrés. Hot-swap propre.

---

## Brancher la base clinique du CHU

À l'hôpital :

1. Implémenter `fetch_new_images_from_clinical_db()` dans
   `backend/app/routers/images.py` (le TODO est balisé).
2. Renseigner `CLINICAL_DB_URL` dans `.env`.
3. Programmer un cron qui appelle `POST /api/images/sync` chaque nuit.
4. Le pont d'IDs (Daytona ↔ consultation) est isolé dans cette fonction —
   c'est le seul point qui change entre dev et prod.

---

## GPU pour le CHU — recommandation

Pour entraînement continu et inférence simultanée sur fonds d'œil HD :

| Option | VRAM | Prix indicatif | Verdict |
|---|---|---|---|
| RTX 4070 Super | 12 GB | ~600 € | Borderline pour gros batch |
| **RTX 4070 Ti Super** | **16 GB** | **~850 €** | **Sweet spot — recommandé** |
| RTX 4080 Super | 16 GB | ~1100 € | +20% plus rapide, marginal |
| RTX 4090 | 24 GB | ~1900 € | Si batch size > 32 ou modèles > 1B |

→ Une RTX 4070 Ti Super tient tranquillement l'inférence continue + les
sessions d'entraînement nocturnes pour 500-5000 images.

---

## Structure du projet

```
retinai-hitl/
├── frontend/               # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/     # layout, queue, viewer, annotation, admin
│   │   ├── pages/          # Login, Annotation, Queue, Admin
│   │   ├── lib/            # store (Zustand), ruleEngine, transcription, mockData
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── Dockerfile
│   └── nginx.conf
├── backend/                # FastAPI
│   ├── app/
│   │   ├── models/         # SQLAlchemy entities (1:1 avec le diagramme de classe)
│   │   ├── schemas/        # Pydantic
│   │   ├── routers/        # auth, images, annotations, catalog, proposals, admin, transcription, model
│   │   ├── core/           # rule_engine, whisper_service
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── seed.py
│   │   └── main.py
│   └── Dockerfile
├── model-service/          # Microservice d'inférence (mock → réel)
│   └── app/main.py
├── docker-compose.yml
└── .env.example
```

---

## Statut & feuille de route

### ✅ Terminé
- Architecture lockée (diagramme de classes v2)
- Frontend complet : login, annotation, file, admin
- Rule engine (urgence + mécanismes) — frontend & backend en miroir
- Catalogue de pathologies (4 chroniques + 5 urgences + 11 rares)
- Grille dynamique (8/16/32/64 cellules selon zoom)
- Dictée FR live (Web Speech API + fallback Whisper)
- Backend skeleton avec tous les endpoints stubbés
- Modèle microservice avec contrat clair pour intégration PyTorch
- Docker Compose multi-services
- Seed du catalogue

### 🔨 À implémenter
- Pont avec base clinique du CHU (fonction balisée TODO)
- Inférence PyTorch réelle (contrat prêt, attendre checkpoint)
- Whisper FR backend (script prêt, attendre GPU)
- Migrations Alembic (actuellement `create_all` au démarrage)
- Tests unitaires backend (pytest)
- Authentification 2FA pour admin

---

## Crédits

Projet de thèse · Service Ophtalmologie CHU · sous la direction de Dr. Moatez Billah Mekki.
Construit avec une approche HITL — l'expertise clinique reste au centre, le modèle apprend.
