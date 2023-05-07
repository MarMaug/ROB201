""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle
from math import dist
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import heapq
import itertools
import tkinter as tk
from tkinter import filedialog, messagebox


class TinySlam:
    """
    Simple occupancy grid SLAM
       
       Summary of the class :
        - Initialisation
        - Convert global coordinates to pixel coordinates
        - Convert pixel coordinates to global coordinates
        - Adding a line and a point on the screen
        - Calculing the score
        - Having access to the global position of the robot
        - Updating the odometry reference point
        - Updating the map 
        - Displaying it
        - Save the map as a png file for example
        
    """
    
    ##################################### INITIALISATION ###################################
    
    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros((int(self.x_max_map), int(self.y_max_map)))
        
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        
        # Initial path from the start to the robot
        self.path = []
        
        #Boolean to indicate if we reached the primary goal
        self.is_primary_goal = 1
        
        # Goal of the robot
        self.goal = np.array([11, -426, np.pi])
        self.click_coords = (self._conv_world_to_map(self.goal[0], -self.goal[1]))
        self.primarygoal = self.goal
        
#########################################################################################################

################################### PARTIE CONVERSION DES COORDONEES ####################################
    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map * self.resolution
        y_world = self.y_min_world + y_map * self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world
    
    
########################################################################################################


###################### PARTIE CLE : TRAITEMENT DE LA POSITION ET DE L'UPDATE DE LA MAP ################

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if (
            x_start < 0
            or x_start >= self.x_max_map
            or y_start < 0
            or y_start >= self.y_max_map
        ):
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(
            np.logical_and(x_px >= 0, x_px < self.x_max_map),
            np.logical_and(y_px >= 0, y_px < self.y_max_map),
        )
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val

    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """

        angles = lidar.get_ray_angles()
        distances = lidar.get_sensor_values()
        theta_world = pose[2]
        score = 0

        # Estimer les positions des détections du laser dans le repère global
        x_world_obs = pose[0] + distances * np.cos(angles + theta_world)
        y_world_obs = pose[1] + distances * np.sin(angles + theta_world)

        # Supprimer les points à la distance maximale du laser 
        # (qui ne correspondent pas à des obstacles)
        indice_bin = np.argwhere(distances < lidar.max_range)
        distances = distances[indice_bin]
        angles = angles[indice_bin]

        # Convertir ces positions dans le repère de la carte
        x_map_obs, y_map_obs = self._conv_world_to_map(x_world_obs, y_world_obs)

        # Supprimer les points hors de la carte
        x_dans_la_map = np.where((0 < x_map_obs) & (x_map_obs < self.x_max_map))
        y_dans_la_map = np.where((0 < y_map_obs) & (y_map_obs < self.y_max_map))
        points_ds_plan = np.intersect1d(x_dans_la_map, y_dans_la_map)
        x_map_obs = x_map_obs[points_ds_plan]
        y_map_obs = y_map_obs[points_ds_plan]

        # Lire et additionner les valeurs des cellule
        # correspondantes dans la carte pour calculer le score
        for i, value in enumerate(x_map_obs):
            score += self.occupancy_map[value, y_map_obs[i]]

        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        odom_pose = np.zeros(3)
        
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        odom_pose[0] = odom[0] + self.odom_pose_ref[0]*np.cos(odom[2] + self.odom_pose_ref[2])
        odom_pose[1] = odom[1] + self.odom_pose_ref[1]*np.sin(odom[2] + self.odom_pose_ref[2])
        odom_pose[2] = odom[2] + self.odom_pose_ref[2]
            
        return odom_pose
    
    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        sigma = 0.1

        N = 0
        # Calculez le score du scan laser avec la position de référence actuelle de l’odométrie (self.odom_pose_ref)
        current_pos = self.get_corrected_pose(odom, self.odom_pose_ref)
        best_score = self.score(lidar, current_pos)

        # répétez tant que moins de N tirages sans amélioration
        while N < 100:
            # Tirez un offset aléatoire selong une gaussienne de moyenne nulle et de variance  et ajoutez le à la position de référence de l’odométrie
            offset = random.gauss(0, sigma)
            new_pos = self.get_corrected_pose(odom, self.odom_pose_ref + offset)

            # Calculez le score du scan laser avec cette nouvelle position de référence de l’odométrie
            new_score = self.score(lidar, new_pos)

            # Si le score est meilleur, mémorisez ce score et la nouvelle position de référence
            if new_score > best_score:
                N = 0
                self.odom_pose_ref = new_pos
                best_score = new_score
            N = N + 1

        return best_score
    
    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """

        """
        Conversion des coordonnées polaires locales des détections du laser (directions/distances)
        à partir de la position absolue du robot en coordonnées cartésiennes absolues dans la carte
        pour avoir les positions des points détectés par le laser
        """
        x_world = pose[0]
        y_world = pose[1]
        theta_world = pose[2]
        angles = lidar.get_ray_angles()
        distances = lidar.get_sensor_values()

        # Supprimer les points à la distance maximale du laser
        # (qui ne correspondent pas à des obstacles)
        indice_range = np.where(distances < lidar.max_range)
        distances = np.array(distances[indice_range])
        angles = np.array(angles[indice_range])

        x_world_obs = x_world + distances * np.cos(angles + theta_world)
        y_world_obs = y_world + distances * np.sin(angles + theta_world)
        # on réduit la longueur de la ligne pour mieux voir apparaître les murs
        x_world_obs_wall = x_world + distances * np.cos(angles + theta_world) * 0.9
        y_world_obs_wall = y_world + distances * np.sin(angles + theta_world) * 0.9

        # Mise à jour de la carte avec chaque point détecté par le
        # télémètre en fonction du modèle probabiliste :
        #    — mise à jour des points sur la ligne entre le robot et
        #      le point détecté avec une probabilité faible
        #    — mise à jour des points détectés par le laser avec une probabilité forte

        val = np.log(0.95 / 0.05)
        # S'arreter avant le point pour faire la ligne pour faire ressortir
        for i in range(len(x_world_obs)):
            self.add_map_line(
                x_world, y_world, x_world_obs_wall[i], y_world_obs_wall[i], -val
            )
        self.add_map_points(x_world_obs, y_world_obs, val + 1)
        
        # Seuillage des probabilités pour éviter les divergences
        self.occupancy_map[self.occupancy_map > 4] = 4
        self.occupancy_map[self.occupancy_map < -4] = -4
        
        # Affichage de la carte mise à jour
        self.display2(pose)
        
#####################################################################################################


######################### PARTIE CALCUL DU CHEMIN DE RETOUR #########################################

    # Version "boostée" de get_neighbors, qui est beaucoup moins chronophage
    def get_neighbors(self, current):
        x, y = current
        neighbors = {(x+i, y+j) for i, j in itertools.product([-1, 0, 1], repeat=2) if (i != 0 or j != 0) and
                 0 <= x+i < self.x_max_map and 0 <= y+j < self.y_max_map and self.occupancy_map[x+i][-(y+j)] < -3}

        return neighbors

    # Def de la distance euclidienne, déjà dans le package math
    def h(self, a, b):
        return dist(a,b)

    # Fonction pour remonter le chemin
    def reconstruct_path(self, cameFrom, current):
        
        # Initialisation du chemin : On commence par le noeud actuel
        total_path = [current]
        
        # Si le noeud actuel est dans cameFrom, ca veut dire qu'il a un prédecesseur
        while current in cameFrom.keys() and cameFrom[current] != None:
            current = cameFrom[current]
            total_path.append(current)
            
        return total_path

    def A_Star(self, start, goal):
        
        # Initialisation
        openSet = [(self.h(start,goal), start)]         # On définit la liste de priorités : de base, c'est juste le start, de fScore h(start, goal)

        # Utiliser un set et non pas une liste est beaucoup plus rapide ( gain de temps : x10)
        visited_nodes = set()
        
        cameFrom = {start: None}             # Le point de départ n'a pas de prédécesseur
        
        # J'ai essayé d'initialiser les dictionnaires avec des valeurs infinies, mais cela s'est révélé extremement couteux en temps
        # J'ai donc intitialisé des dictionnaires "vides" et on compare les valeurs de gScore seulement quand il y a une valeur à comparer
        # C'est à dire quand le noeud a déjà été visité.
        
        gScore= {start : 0}                     # Distance start -> start : 0
        fScore = {start : self.h(start, goal)}  # Distance start -> goal : h(start, goal)
        
        # Tant que l'on a des noeuds à explorer
        while openSet is not []:
            # On pop le noeud avec le fScore le plus faible
            current = heapq.heappop(openSet)[1]
            
            # Si c'est le goal, banco
            if current == goal:
                path = self.reconstruct_path(cameFrom, current)
                return path
            
            # Sinon, on regarde ses voisins
            visited_nodes.add(current)
            neighbors = self.get_neighbors(current)

            # Pour chacun des voisins
            for neighbor in neighbors:
                
                    # On regarde son gScore a travers le noeud actuel
                    tentative_gScore = gScore[current] + self.h(current, neighbor)

                    # Si le voisin a déjà été visité et si la tentative de gScore n'améliore rien, on passe
                    if neighbor in visited_nodes and tentative_gScore >= gScore[neighbor] - 10e-3:
                        continue
                    
                    # Sinon
                    cameFrom[neighbor] = current                                 # On note le voisin comme issu du noeud courant
                    gScore[neighbor] = tentative_gScore                          # On actualise le gScore de ce voisin
                    fScore[neighbor] = tentative_gScore + self.h(neighbor, goal) # On note le fScore de ce voisin
                    if ((fScore[neighbor], neighbor) not in openSet):
                        # On rajoute ce voisin dans openSet, avec son fScore
                        heapq.heappush(openSet, (fScore[neighbor], neighbor))
        
        print("Echec de l'algorithme")
        return None


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """

        # convert from world to map coordinates
        start = self._conv_world_to_map(start[0], -start[1])
        goal  = self._conv_world_to_map(goal[0], -goal[1])

        path = self.A_Star(start, goal)
        
        return path

#########################################################################################


############################ PARTIE AFFICHAGE ECRAN ET SAUVEGARDE #######################

    # fonction appelée lorsqu'un clic de souris est détecté
    def detect_click(self,event, x, y, flags,param):
    # si l'utilisateur a cliqué avec le bouton gauche de la souris
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # Si l'emplacement est valide
            if self.occupancy_map[x,-y] <3.0:
                
                # Le goal redevient primaire
                self.is_primary_goal = 1
                
                # J'actualise le goal selon les coordonées du clic de la souris
                self.goal = np.array([self._conv_map_to_world(x,y)[0],
                                -self._conv_map_to_world(x,y)[1], np.pi])
                
                # Je note le goal actuel comme étant primaire
                self.primarygoal = self.goal
            else :
                
                print("The selected goal is not reachable by the robot")
                
            self.click_coords = (x, y)
        
        
    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        cv2.namedWindow("map slam")
        
        
        # Detection d'un clic
        cv2.setMouseCallback("map slam", self.detect_click)
        
        # Dessiner la fleche rouge du robot
        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)
        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)
        
        # Dessiner le chemin retour vers le start
        for node in self.path:
            pt_node = (node[0], node[1])
            cv2.circle(img2, pt_node, 0, color=(0, 0, 255), thickness=-1)
        
        # Ecrire le texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        org2 = (50, 70)
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        
        cv2.putText(img2, "Robot position (" + str(round(robot_pose[0])) + ", " + str(round(robot_pose[1])) + ")", org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        cv2.putText(img2, "Goal position : (" + str(round(self.goal[0])) + ", " + str(round(self.goal[1])) + ")", org2, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        type = ''
        if self.occupancy_map[self.click_coords[0],-self.click_coords[1]] >3.0:
            type = 'Obstacle'
        elif self.occupancy_map[self.click_coords[0],-self.click_coords[1]] <-3.0:
            type = 'Ground'
        else:
            type = "Out of bounds"
            
        cv2.putText(img2, "Type of the last selected point : " + type , (50,110), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        pos_rob = (robot_pose[0], robot_pose[1])
        
        primarygoal = self._conv_world_to_map(self.primarygoal[0], -self.primarygoal[1])
        
        # Indiquer si on a atteint le goal primaire
        if dist(pos_rob, [self.primarygoal[0], self.primarygoal[1]]) <= 10:
            cv2.putText(img2, "Goal reached !", (50,90), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)

        # Dessiner un cercle blanc à l'emplacement du goal primaire
        cv2.circle(img2, primarygoal, 2, color=(255, 255, 255), thickness=-1)    
        
        # Dessiner un cercle vert au start
        cv2.circle(img2, self._conv_world_to_map(0,0), 3, color=(0, 255, 0), thickness=-1)
     
        # Show image
        cv2.imshow("map slam", img2)
        key = cv2.waitKey(1)
        
        # Si la touche "s" est appuyée, exécuter l'action souhaitée
        if key == ord('s'):
            
            root = tk.Tk()
            root.withdraw()
            
            # Vérifier si l'utilisateur a confirmé la sauvegarde
            confirmation = messagebox.askyesno("Confirmation", "Êtes-vous sûr de vouloir sauvegarder l'image ?")
            
            if confirmation:
                chemin_destination = filedialog.asksaveasfilename(defaultextension='PNG', filetypes=[('PNG', '.png'), ('JPEG', '.jpg')])
       
                # Enregistrer l'image  
                cv2.imwrite(chemin_destination, img2)

        # Si la touche "ESC" est appuyée, quitter la boucle
        if key == 27 or key == ord('q'):
            exit()
        
        
    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(
            self.occupancy_map.T,
            origin="lower",
            extent=[
                self.x_min_world,
                self.x_max_world,
                self.y_min_world,
                self.y_max_world,
            ],
        )
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + ".png")

        with open(filename + ".p", "wb") as fid:
            pickle.dump(
                {
                    "occupancy_map": self.occupancy_map,
                    "resolution": self.resolution,
                    "x_min_world": self.x_min_world,
                    "x_max_world": self.x_max_world,
                    "y_min_world": self.y_min_world,
                    "y_max_world": self.y_max_world,
                },
                fid,
            )
