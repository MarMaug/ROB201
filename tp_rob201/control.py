""" A set of robotics control functions """
import random as rd
from math import dist
import numpy as np


# Fonction obsolete
def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    angles = lidar.get_ray_angles()
    index = np.where(angles == 0)[0][0]
    distances = lidar.get_sensor_values()
    maximum = np.max(distances)
    rotation = rd.uniform(-1, 1)

    if distances[index] < 50:
        command = {"forward": -1, "rotation": rotation}

    else:
        command = {"forward": 1, "rotation": 0}

    return command


# Vraie fonction de recherche du goal
def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    # A twister selon les usages
    d_change = 50
    r_min = 10
    d_safe = 20

    # On récupère les données 
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    position = np.array([pose[0], pose[1]])
    but = np.array([goal[0], goal[1]])
    ecart = but - position
    ecart_norm = dist(pose,goal)
    
    #Détection de l'obstacle le plus proche
    index = np.argmin(distances)
    mindist = distances[index]
    minangle = angles[index]
    
    # On en déduit sa position
    obstacle_position = np.array([pose[0] + mindist*np.cos(minangle+pose[2]), pose[1] + mindist*np.sin(minangle+pose[2])])
    
    # Si je suis en dessous de la d_safe, je calcule le gradient d'evitement d'obstacle
    if mindist < d_safe :
        Kobs = 5000
        gradient_obstacle = Kobs/(mindist**3)*((1/mindist)-(1/d_safe))*(obstacle_position - np.array([pose[0],pose[1]]))
    else :
        gradient_obstacle = np.array([0,0])

    #Cas éloigné - Potentiel conique.
    if ecart_norm > d_change :
        
        K_cone = 0.5
        gradient = np.array([(K_cone/ecart_norm)*ecart[0], (K_cone/ecart_norm)*ecart[1]])
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        rotation = np.clip(gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)

    #Cas proche - Potentiel quadratique.
    elif r_min < ecart_norm <= d_change :
        
        #print("Approaching the goal")
        Kquad = 0.2/d_change
        gradient = np.array([Kquad*ecart[0], Kquad*ecart[1]])
        #print("Old Gradient : ", gradient)
        gradient = gradient - gradient_obstacle
        #print("New Gradient : ", gradient)
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(1.5*gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        #velocity = np.clip(gradient_norme, -1, 1)
        rotation = np.clip(3*gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)
    
    #Cas touché - On s'arrête.
    elif ecart_norm <= r_min :
        velocity = 0
        rotation = 0
        
    command = {"forward": velocity, "rotation": rotation}
    return command