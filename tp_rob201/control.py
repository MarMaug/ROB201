""" A set of robotics control functions """
import random as rd

import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    angles = lidar.get_ray_angles()
    index = np.where(angles == 0)[0][0]
    distances = lidar.get_sensor_values()
    maximum = np.max(distances)
    print(distances, maximum)
    rotation = rd.uniform(-1, 1)

    if distances[index] < 50:
        command = {"forward": -1, "rotation": rotation}

    else:
        command = {"forward": 1, "rotation": 0}

    return command


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    
    k_cone = 0.1
    d_safe = 200
    k_obs = 1000
    d_chang = 30
    r_valid = 10
    k_quad = k_cone / d_chang

    # Conversion du goal dans le repère du robot
    # goal_x_rob = goal[0]*np.cos(theta_rob)+goal[1]*np.sin(theta_rob)
    # goal_y_rob = -goal[0]*np.sin(theta_rob)+goal[1]*np.cos(theta_rob)
    # goal_angle = goal[2]-theta_rob
    # goal_rob = np.array([goal_x_rob,goal_y_rob,goal_angle])

    # Calcul du gradient objectif
    # Commencez par récupérer vos données LIDAR (lidar.get_sensor_values() & lidar.get_ray_angles())
    # et extrayez l'indice correspondant à la distance la plus courte
    angles = lidar.get_ray_angles()
    distances = lidar.get_sensor_values()
    indice_min = np.argmin(distances)
    dmin = distances[indice_min]
    angle_min = angles[indice_min]

    # Calcul du gradient répulsif
    # Calculez votre vecteur répulsif selon la formule proposée en cours
    if dmin < d_safe:
        grad_rep = (
            (k_obs / dmin**3)
            * ((1 / dmin) - (1 / d_safe))
            * (dmin * np.array([np.cos(angle_min), np.sin(angle_min)]))
        )
    else:
        grad_rep = np.array([0, 0])

    # calcul de la distance à l'objectif et la direction de l'objectif dans le repère local du robot
    position = np.array([pose[0], pose[1]])
    but = np.array(goal[0], goal[1])
    ecart = but - position
    d = np.linalg.norm(ecart)
    # expression du gradient attractif dans le repère robot à partir de l'expression
    # de la position goal dans le repère du robot

    # potentiel conique
    if d > d_chang:
        grad_obj = (k_cone / d) * np.array([ecart[0], ecart[1]])
    elif r_valid < d <= d_chang:
        # potentiel quadratique
        grad_obj = k_quad * np.array([ecart[0], ecart[1]])
    else:
        # on est arrivé
        command = {"forward": 0, "rotation": 0}
        return command

    grad_pose = grad_obj - grad_rep

    # Normalisez
    grad_angle_norm = np.arctan2(grad_pose[1], grad_pose[0]) / np.pi
    grad_rep_norm = np.clip(grad_pose, -1, 1)

    # Déduire une commande
    if np.abs(grad_angle_norm) > np.pi / 2:
        if grad_angle_norm > 0:
            command = {"forward": 0, "rotation": 1}
        else:
            command = {"forward": 0, "rotation": -1}
    else:
        forward = np.clip(
            np.linalg.norm(grad_rep_norm) * np.cos(grad_angle_norm), -1, 1
        )
        rotation = np.clip(
            np.linalg.norm(grad_rep_norm) * np.sin(grad_angle_norm), -1, 1
        )
        command = {"forward": forward, "rotation": rotation}

    # Ajustez vos paramètres

    return command
