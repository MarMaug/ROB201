"""
This file was generated by the tool 'image_to_map.py' in the directory tools.
This tool permits to create this kind of file by providing it an image of the map we want to create.
"""

from place_bot.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (1113,750)
# Dimension factor : 1.0
def add_boxes(playground):
    # box 0 en bas à droite
    box = NormalBox(up_left_point=(-556.5, -222.0),
                    width=437, height=153)
    playground.add(box, box.wall_coordinates)

    # box 1
    box = NormalBox(up_left_point=(6.5, 69.0),
                    width=130, height=100)
    playground.add(box, box.wall_coordinates)

    # box 3 en haut à gauche
    box = NormalBox(up_left_point=(-556.5, 375.0),
                    width=434, height=155)
    playground.add(box, box.wall_coordinates)


def add_walls(playground):
    # vertical wall 0
    wall = NormalWall(pos_start=(-128.5, 362.0),
                      pos_end=(-128.5, 222.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 1
    wall = NormalWall(pos_start=(-127.5, 361.0),
                      pos_end=(548.5, 361.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 2
    wall = NormalWall(pos_start=(546.5, 360.0),
                      pos_end=(546.5, -364.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 3
    wall = NormalWall(pos_start=(147.5, 361.0),
                      pos_end=(147.5, 226.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 4
    wall = NormalWall(pos_start=(149.5, 361.0),
                      pos_end=(149.5, 226.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 5
    wall = NormalWall(pos_start=(335.5, 361.0),
                      pos_end=(335.5, 258.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(337.5, 360.0),
                      pos_end=(337.5, 258.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 10
    wall = NormalWall(pos_start=(-543.5, 227.0),
                      pos_end=(-543.5, -226.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 11
    wall = NormalWall(pos_start=(-542.5, 226.0),
                      pos_end=(-127.5, 226.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 12
    wall = NormalWall(pos_start=(-400.5, 226.0),
                      pos_end=(-400.5, -36.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 13
    wall = NormalWall(pos_start=(-398.5, 226.0),
                      pos_end=(-398.5, -36.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 14
    wall = NormalWall(pos_start=(337.5, 200.0),
                      pos_end=(337.5, -111.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 15
    wall = NormalWall(pos_start=(335.5, 199.0),
                      pos_end=(335.5, -111.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 17
    wall = NormalWall(pos_start=(149.5, 161.0),
                      pos_end=(149.5, 137.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 18
    wall = NormalWall(pos_start=(150.5, 139.0),
                      pos_end=(336.5, 139.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 19
    wall = NormalWall(pos_start=(147.5, 160.0),
                      pos_end=(147.5, 134.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 20
    wall = NormalWall(pos_start=(148.5, 137.0),
                      pos_end=(336.5, 137.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 21
    wall = NormalWall(pos_start=(-302.5, 138.0),
                      pos_end=(-302.5, 98.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 22
    wall = NormalWall(pos_start=(-302.5, 136.0),
                      pos_end=(-125.5, 136.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 23
    wall = NormalWall(pos_start=(-300.5, 136.0),
                      pos_end=(-300.5, 98.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 24
    wall = NormalWall(pos_start=(-301.5, 134.0),
                      pos_end=(-128.5, 134.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 25
    wall = NormalWall(pos_start=(-130.5, 135.0),
                      pos_end=(-130.5, -226.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 26
    wall = NormalWall(pos_start=(-543.5, 114.0),
                      pos_end=(-525.5, 114.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 27
    wall = NormalWall(pos_start=(-435.5, 114.0),
                      pos_end=(-399.5, 114.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 28
    wall = NormalWall(pos_start=(-543.5, 112.0),
                      pos_end=(-526.5, 112.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 29
    wall = NormalWall(pos_start=(-436.5, 112.0),
                      pos_end=(-399.5, 112.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 30
    wall = NormalWall(pos_start=(8.5, 68.0),
                      pos_end=(8.5, -27.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 31
    wall = NormalWall(pos_start=(9.5, 67.0),
                      pos_end=(132.5, 67.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 32
    wall = NormalWall(pos_start=(130.5, 67.0),
                      pos_end=(130.5, -27.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 33
    wall = NormalWall(pos_start=(-300.5, 12.0),
                      pos_end=(-300.5, -226.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 34
    wall = NormalWall(pos_start=(-302.5, 11.0),
                      pos_end=(-302.5, -226.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 35
    wall = NormalWall(pos_start=(8.5, -25.0),
                      pos_end=(131.5, -25.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 36
    wall = NormalWall(pos_start=(-129.5, -107.0),
                      pos_end=(205.5, -107.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 37
    wall = NormalWall(pos_start=(277.5, -107.0),
                      pos_end=(336.5, -107.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 38
    wall = NormalWall(pos_start=(36.5, -107.0),
                      pos_end=(36.5, -277.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 39
    wall = NormalWall(pos_start=(-128.5, 136.0),
                      pos_end=(-128.5, -227.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 40
    wall = NormalWall(pos_start=(-125.5, -224.0),
                      pos_end=(-125.5, -364.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 41
    wall = NormalWall(pos_start=(-125.5, -362.0),
                      pos_end=(547.5, -362.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 42
    wall = NormalWall(pos_start=(-129.5, -109.0),
                      pos_end=(205.5, -109.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 43
    wall = NormalWall(pos_start=(276.5, -109.0),
                      pos_end=(336.5, -109.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 44
    wall = NormalWall(pos_start=(34.5, -108.0),
                      pos_end=(34.5, -277.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 45
    wall = NormalWall(pos_start=(-398.5, -116.0),
                      pos_end=(-398.5, -154.0))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 46
    wall = NormalWall(pos_start=(-400.5, -117.0),
                      pos_end=(-400.5, -154.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 47
    wall = NormalWall(pos_start=(-543.5, -142.0),
                      pos_end=(-399.5, -142.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 48
    wall = NormalWall(pos_start=(-543.5, -144.0),
                      pos_end=(-399.5, -144.0))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 49
    wall = NormalWall(pos_start=(-543.5, -224.0),
                      pos_end=(-129.5, -224.0))
    playground.add(wall, wall.wall_coordinates)

