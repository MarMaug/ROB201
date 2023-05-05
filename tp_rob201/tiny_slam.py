""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle
from math import dist
from math import pi
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import heapq

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world
        )

        self.occupancy_map = np.zeros((int(self.x_max_map), int(self.y_max_map)))
        self.counter = 0
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        self.path = []

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
        odom_pose[1] = odom[1] + self.odom_pose_ref[0]*np.sin(odom[2] + self.odom_pose_ref[2])
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
        # Initial position
        init_coord = np.array([0,0,0])
        if self.counter % 100 == 0:
            self.path = self.plan(init_coord,pose )
            print("chemin mis à jour")
        self.counter += 1
        # Seuillage des probabilités pour éviter les divergences
        self.occupancy_map[self.occupancy_map > 4] = 4
        self.occupancy_map[self.occupancy_map < -4] = -4
        self.display2(pose, self.path)

#########################################################################################

    def get_neighbors(self, current):
        x, y = current
        neighbors = []

        for i in [-1, 0, +1]:
            for j in [-1, 0, +1]:
                if i == 0 and j == 0:
                    continue

                new_x, new_y = x+i, y+j
                if (new_x < 0 or new_x >= self.x_max_map or
                    new_y < 0 or new_y >= self.y_max_map):
                    continue

                neighbors.append((new_x, new_y))

        return neighbors

    # Def de la distance euclidienne, déjà dans le package math
    def h(self, a, b):
        x_a, y_a = a
        x_b, y_b = b

        # euclidean distance
        return np.sqrt( (x_b - x_a)**2 + (y_b - y_a)**2 )

    # Fonction pour remonter le chemin
    def reconstruct_path(self, cameFrom, current_id, current):
        
        # Initialisation du chemin : On commence par le noeud actuel
        total_path = [current]
        
        # Si le noeud actuel est dans cameFrom, ca veut dire qu'il a un prédecesseur
        while current_id in cameFrom.keys():
            current = cameFrom[tuple(current)]
            np.append(total_path, current)
        return total_path


    def A_Star(self, start, goal):
        #Initialisation
        
        openSet = [tuple(start)]           # 1er point observé : le start
        
        cameFrom = {}               # Noeud juste avt le départ : aucun (c'est normal) 
        
        gScore = defaultdict(lambda: 10e7)
        
        gScore[tuple(start)] = 0      # Meilleur score connu entre le start et notre noeud initial : 0
        
        visited_nodes = []
        
        fScore = gScore        
        fScore[tuple(start)] = self.h(start, goal) # Meilleur score connu entre le start et le goal : dist(start, goal)
        
        # Tant que l'on a des noeuds découverts à explorer
        while openSet != []:

            # On selectionne le noeud le plus proche de l'arrivée
            current = openSet[np.argmin(fScore)]
            visited_nodes.append(current)
            
            # Si c'est le noeud d'arrivée : Top !
            if current == goal:
                return self.reconstruct_path(cameFrom, current)
            
            # Sinon : 
            openSet.remove(current)                     # On note le noeud comme étant exploré
            voisins = self.get_neighbors(current)       # On regarde ses voisins
            # Pour tous ses voisins :    
            for i, ce_voisin in enumerate(voisins):

                # On regarde la distance entre le start et ce voisin à travers le noeud courant
                tentative_gScore = gScore[tuple(current)] + self.h(current, ce_voisin)
                
                # Si ce chemin start->voisin est meilleur qu'un autre déjà enregistré, on le met à jour
                if (tentative_gScore < gScore[tuple(ce_voisin)]):
                    cameFrom[tuple(ce_voisin)] = current
                    gScore[tuple(ce_voisin)] = tentative_gScore
                    fScore[tuple(ce_voisin)] = tentative_gScore + self.h(ce_voisin, goal)  # On met aussi à jour le fScore
                    
                    # On ajoute le voisin dans les noeuds à étudier si ce n'est pas déjà le cas
                    if ce_voisin not in openSet:
                        if ce_voisin not in visited_nodes:
                            openSet.append(ce_voisin)
            
        print("Echec")
        return []

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """

        # convert from world to map coordinates
        start = self._conv_world_to_map(start[0], -start[1])
        goal  = self._conv_world_to_map(goal[0], -goal[1])

        # print(f'start: {start} goal: {goal}')

        # heapq initialization
        priority_heap = [(0, start)]
        visited_nodes = set()

        # 
        parent_nodes = {start: None}

        # scores dictionaries
        g_values = {start: 0}
        f_values = {start: self.h(start, goal)}

        # print('plan')
# 
        # loop util heap is empty, meaning: no path found or goal found
        while priority_heap:
            # print('while')
            # pop node with smallest f value
            _, current_node = heapq.heappop(priority_heap)

            # print(f'current_node: {current_node}')

            # if goal was found, reconstruct path and return it
            if current_node == goal:
                path = []

                while current_node is not None:
                    # print('path')
                    path.append(current_node)
                    current_node = parent_nodes[current_node]

                # return reverse path
                return path[::-1]

            # goal not found, store current node as visited and continue searching
            visited_nodes.add(current_node)

            neighbors = self.get_neighbors(current_node)
            # print(f'neighbors: {neighbors}')

            # loop through possible movements
            for neighbor in neighbors:
                # print(f'neighbor: {neighbor}')
                # print('neighbor')

                x_valid = 0 <= neighbor[0] < self.x_max_map
                y_valid = 0 <= neighbor[1] < self.y_max_map
                notObstacule = self.occupancy_map[neighbor[0]][neighbor[1]] < 0.75*4

                #print(f'{x_valid} {y_valid} {notObstacule}')

                # check if neighbor is inside bounders and is not an obstacle
                if x_valid and y_valid and notObstacule:
                    # compute neighbor's g value
                    tentative_g_value = g_values[current_node] + self.h(current_node, neighbor)

                    # print(f'tentative_g_value: {tentative_g_value}')
                    # print(f'visited_nodes: {visited_nodes}')
                    # print(f'g_values: {g_values.get(neighbor, 0)}')

                    # 
                    if neighbor in visited_nodes and tentative_g_value >= g_values.get(neighbor, 0):
                        continue

                    parent_nodes[neighbor] = current_node
                    g_values[neighbor] = tentative_g_value
                    f_values[neighbor] = tentative_g_value + self.h(neighbor, goal)

                    heapq.heappush(priority_heap, (f_values[neighbor], neighbor))

            # break

        # if heap is empty and no path was found
        return None
#########################################################################################

    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
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

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(
            robot_pose[0],
            robot_pose[1],
            delta_x,
            delta_y,
            color="red",
            head_width=5,
            head_length=10,
        )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose, path):
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

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        #print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=2)
        
        for node in path:
            pt_node = (node[0], node[1])
            cv2.circle(img2, pt_node, 1, color=(0, 0, 255), thickness=2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        org2 = (50, 70)
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        cv2.putText(img2, "x pos : " + str(round(robot_pose[0])), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img2, "y pos : " + str(round(robot_pose[1])), org2, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        goal = self._conv_world_to_map(70, -120)
        cv2.circle(img2, goal, 2, color=(255, 255, 255), thickness=3)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

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

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO

        # def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

    #    ranges = np.random.rand(3600)
    #    ray_angles = np.arange(-np.pi,np.pi,np.pi/1800)

    # Poor implementation of polar to cartesian conversion
    #    points = []
    #    for i in range(3600):
    #       pt_x = ranges[i] * np.cos(ray_angles[i])
    #       pt_y = ranges[i] * np.sin(ray_angles[i])
    #       points.append([pt_x,pt_y])
