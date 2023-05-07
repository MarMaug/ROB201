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
          
        # Counter to follow the path back to the start
        self.counter = 0
        
    def control(self):
        """
        Main control function executed at each time step
        """
      
        # Localising and exploring
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), self.tiny_slam.get_corrected_pose(self.odometer_values(), None), self.tiny_slam.goal)

        # Si on est arrivé
        if dist(self.tiny_slam.get_corrected_pose(self.odometer_values(), None), self.tiny_slam.goal) <=10:
            
            if self.tiny_slam.counter == 1:
                start = time()
                self.tiny_slam.path = self.tiny_slam.plan(np.array([0, 0, 0]), self.tiny_slam.get_corrected_pose(self.odometer_values(), None) )
                print("Temps de calcul du chemin : ", round(time() - start,3), " s.")
                self.tiny_slam.counter = 0
                self.counter = 0
            else:
                #print("Trop proche du nouveau goal : on actualise")
                if self.counter < len(self.tiny_slam.path)-3:
                    self.counter+=3
                    self.tiny_slam.goal = np.array([self.tiny_slam._conv_map_to_world(self.tiny_slam.path[self.counter][0],self.tiny_slam.path[self.counter][1])[0],
                              - self.tiny_slam._conv_map_to_world(self.tiny_slam.path[self.counter][0],self.tiny_slam.path[self.counter][1])[1], np.pi])
                
        return command 