# Inventaire initial des robots de trading

Le catalogue associé, [`robot_inventory.csv`](robot_inventory.csv), recense les fichiers repérés dans `Téléchargements/Telegram Desktop` le 15 octobre 2026.

La matrice [`robot_test_matrix.csv`](robot_test_matrix.csv) suit les 114 exécutables `.ex4`/`.ex5` individuellement. Les indicateurs de risque extraits des noms sont de simples alertes à vérifier, pas une analyse du code ou du comportement.

## Contenu recensé

- 106 fichiers `.ex5` pour MT5
- 8 fichiers `.ex4` pour MT4
- 9 archives liées aux robots ou à leurs presets, dont 8 archives ZIP de presets et `Gold Nemesis.ex5.rar`
- 1 fichier de paramètres `.set`

Les noms et extensions ne suffisent pas à confirmer qu'un fichier est un robot de trading. Certains peuvent être des indicateurs ou des outils, et les variantes de version ou de langue peuvent être des doublons. Les colonnes de type et de marché sont donc des hypothèses à vérifier.

## Limites de cet inventaire

- Seuls les noms, extensions, tailles et dates des fichiers ont été relevés. Les exécutables et archives n'ont pas été ouverts ni lancés.
- Aucun fichier source `.mq4` ou `.mq5` n'a été repéré dans ce dossier lors de la recherche initiale. Les `.ex4` et `.ex5` sont des fichiers compilés.
- Les archives de presets et l'archive `Gold Nemesis.ex5.rar` restent à inspecter avant de connaître leur contenu exact.
- Un `.ex4` relève de MT4 et un `.ex5` de MT5 ; il faudra valider les robots dans leur plateforme correspondante.

## Étapes de validation

1. Confirmer pour chaque entrée s'il s'agit d'un expert, d'un indicateur, d'un outil ou d'un preset.
2. Rassembler la documentation, les paramètres et les dépendances de chaque robot.
3. Regrouper les variantes et doublons possibles, puis sélectionner les robots testables.
4. Faire les tests dans un environnement de démonstration, avec coûts réalistes et périodes distinctes de réglage et de validation.
5. Comparer le drawdown, le profit factor, l'espérance par transaction, l'exposition et la stabilité, en plus du profit net.

Le catalogue ne contient pas les exécutables eux-mêmes et n'implique pas qu'ils aient été testés.

## Préparation des simulations

Le poste a MT5 installé et déjà ouvert ; MT4 n'a pas été trouvé dans les emplacements usuels. Six fichiers `.ex5` du catalogue apparaissent déjà dans le dossier Experts du profil MT5. Le cache du dépôt contient des barres journalières pour plusieurs instruments, mais pas de données XAUUSD. Le cache historique MT5 repéré contient des données EURUSD ; aucune série intrajournalière XAUUSD n'a été trouvée.

Les robots ne sont donc pas encore testés. Avant les simulations, il faut disposer d'historiques adaptés aux instruments et unités de temps de chaque robot. Les séries journalières existantes ne permettent pas de représenter correctement un robot de scalping ou intrajournalier.
