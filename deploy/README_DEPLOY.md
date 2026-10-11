# Deploiement Rex Clemens Capital

## Option 1 : PythonAnywhere (le plus simple)

**Gratuit** — 1 trade/jour, dashboard web, zero config serveur.

### Etapes

1. Creer un compte sur **pythonanywhere.com** (gratuit)
2. Aller dans **Dashboard > Consoles > Bash**
3. Cloner le repo :
   ```
   git clone https://github.com/ashbornarise/Azazel-ABBADON.git
   cd Azazel-ABBADON
   pip install --user pandas numpy scikit-learn yfinance flask
   ```
4. Creer l'app web :
   - Aller dans **Web** > **Add a new web app**
   - Choisir **Flask** + **Python 3.10**
   - Path du code source : `/home/TONPSEUDO/Azazel-ABBADON`
   - Fichier WSGI : remplacer le contenu par celui de `deploy/pa_wsgi.py`
5. Tache planifiee (1 trade/jour) :
   - Aller dans **Tasks**
   - Ajouter : `cd ~/Azazel-ABBADON && python run_daily.py`
   - Heure : 21:00 UTC (23h Paris)
6. Recharger l'app web → le dashboard est en ligne !

URL : `https://TONPSEUDO.pythonanywhere.com`

### Limites du gratuit
- 1 app web
- 1 tache planifiee par jour (parfait pour 1 trade/jour)
- CPU limitee (~100 sec/jour)
- Internet sortant : whitelist seulement (yfinance marche)


## Option 2 : Google Cloud (VM gratuite 24/7)

**Gratuit pour toujours** — VM e2-micro, dashboard live, robot en continu.

### Etapes

1. Creer un compte sur **cloud.google.com** (gratuit, carte demandee mais pas debitee)
2. Aller dans **Compute Engine > VM instances > Create**
   - Nom : `rex-clemens`
   - Region : `us-central1` (gratuit)
   - Machine : `e2-micro` (gratuit)
   - OS : Ubuntu 22.04
   - Disque : 30 Go (gratuit)
   - Cocher "Allow HTTP traffic"
3. Se connecter en SSH (bouton "SSH" dans la console)
4. Lancer le script d'installation :
   ```
   git clone https://github.com/ashbornarise/Azazel-ABBADON.git
   cd Azazel-ABBADON
   bash deploy/install_gcloud.sh
   ```
5. Le robot tourne, le dashboard est sur `http://IP_DU_SERVEUR:8777`

### Limites du gratuit
- 1 VM e2-micro (2 vCPU, 1 GB RAM)
- 30 Go disque
- Region us-central1 uniquement
- 1 Go sortant/mois


## Securite (IMPORTANT)

- Aucun argent reel sur le serveur
- Pas de cle API de courtier
- Le robot est en mode SIMULATION uniquement
- Phase 5 (argent reel) = decision de Clement, jamais automatique
