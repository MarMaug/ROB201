"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""

import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams
from tiny_slam import TinySlam
from control import potential_field_control
from math import dist
from time import time


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(
        self,
        lidar_params: LidarParams = LidarParams(),
        odometer_params: OdometerParams = OdometerParams(),
    ):
        # Passing parameter to parent class
        super().__init__(
            should_display_lidar=False,
            lidar_params=lidar_params,
            odometer_params=odometer_params,
        )
        
        # Init SLAM object
        self._size_area = (1113, 750)
        self.tiny_slam = TinySlam(
            x_min=-self._size_area[0],
            x_max=self._size_area[0],
            y_min=-self._size_area[1],
            y_max=self._size_area[1],
            resolution=2,
        )

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
          
        # Counter to follow where we are in the path back to the start
        self.counter_back_to_start = 0
    
    
    def control(self):
        """
        Main control function executed at each time step
        At each time step, 
            - We update the odometry, the lidar and the map
            - We calculate the command to give to the robot to reach the goal (may it be the primary one or not)
            - If we reached a goal, 2 possible cases :
                    
                    - If we reached the "primary" goal, displayed in white on the map (indicated by the boolean "is_primary_goal"):
                        We calculate the shortest path from our start point and we indicate we reached the goal
                        We prepare to go back to the start, by setting our counter "back_to_start" to 0
                    
                    - If we reached an intermediate goal while coming down the path:
                        We go to the next goal down the path (here, it's 3 steps further)

            - We return the command
        """
      
        # Localising and exploring
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), self.tiny_slam.get_corrected_pose(self.odometer_values(), None), self.tiny_slam.goal)

        # Si on est arrivé
        if dist(self.tiny_slam.get_corrected_pose(self.odometer_values(), None), self.tiny_slam.goal) <=10:
            
            # Si on est au goal primaire (celui de base, ou celui indiqué par un clic de souris)
            if self.tiny_slam.is_primary_goal:
                
                # On calcule le chemin retour le plus court grace à A*
                start = time()
                self.tiny_slam.path = self.tiny_slam.plan(np.array([0, 0, 0]), self.tiny_slam.get_corrected_pose(self.odometer_values(), None) )
                print("Temps de calcul du chemin : ", round(time()- start,3), " s.")
                print(self.tiny_slam.path)
                # On dit qu'on a atteint le goal primaire en mettant le booleen 'is_primary_goal' à 0
                self.tiny_slam.is_primary_goal = 0
                
                # On réinitialise le compteur de retour à 0 pour dire qu'on repart au début du chemin de retour
                self.counter_back_to_start = 0
            
            else:
                #print("Arrivé au goal intermediaire : on va au suivant")
                
                # Tant qu'on est dans le path de retour
                if self.counter_back_to_start < len(self.tiny_slam.path)-3:
                    
                    x_converti, y_converti = self.tiny_slam._conv_map_to_world(self.tiny_slam.path[self.counter_back_to_start][0],self.tiny_slam.path[self.counter_back_to_start][1])
                    # On update le goal à 3 itérations plus loin
                    self.counter_back_to_start +=3
                    self.tiny_slam.goal = np.array([x_converti, -y_converti, np.pi])
                    
                    # Dés qu'on sera arrivé, ca refera une itération, etc ...
                    
        return command 