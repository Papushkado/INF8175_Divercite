# Vue d'ensemble de GameStateDivercite

GameStateDivercite est une classe qui hérite de GameState et représente l'état d'une partie du jeu Divercite. Voici les principales fonctionnalités et méthodes :
## Attributs

`scores:` Liste des scores des joueurs.
`next_player:` Prochain joueur à jouer.
`players:` Liste des joueurs.
`rep:` Représentation actuelle du plateau de jeu.
`max_step:` Nombre maximum de tours (40).
`step:` Étape actuelle du jeu.
`players_pieces_left:` Nombre de pièces restantes pour chaque joueur.

## Méthodes Principales

`Constructeur :`n Initialise les scores, le joueur suivant, la liste des joueurs, la représentation du plateau, le nombre d'étapes, et le nombre de pièces restantes.
`get_step :` Retourne le nombre d'étapes actuelles.
`is_done :` Vérifie si le jeu est terminé.
`get_neighbours :` Récupère les voisins d'une position donnée sur le plateau.
`in_board :` Vérifie si un index donné est dans le plateau.
`piece_type_match :` Vérifie si une pièce peut être placée à une position donnée.
`get_player_id :` Récupère le joueur par son identifiant.
`generate_possible_heavy_actions` et `generate_possible_light_actions :` Génére des actions possibles pour le joueur suivant.
`apply_action :` Applique une action légère à l'état du jeu.
`convert_gui_data_to_action_data :` Convertit les données de l'interface graphique en données d'action.
`compute_players_pieces_left` et `compute_scores :` Calculent les pièces restantes et les scores respectivement.
`check_divercite :` Vérifie si une position a gagné une divercite.
`remove_draw :` Gère les cas de match nul en déterminant un gagnant basé sur des critères supplémentaires.
`to_json` et `from_json :` Gère la sérialisation et la désérialisation de l'état du jeu.