import math
import cv2
import numpy as np
from random import randint


class Table:
    def __init__(self, _image):
        self.image = _image
        self.balls = []
        self.pockets = []

    # show the table, including marked balls, pockets and best shot for a ball of type ball_type (eg solids)
    def show_best_shot(self, ball_type):
        output = self.image

        for pocket in self.pockets:
            # show a red circle for each pocket
            cv2.circle(output, pocket.position.tuple(), pocket.radius, Colour.POCKET, 2)

        for ball in self.balls:
            if ball.type == "solids":
                colour = Colour.SOLIDS  # show a blue circle for a solids ball
            elif ball.type == "stripes":
                colour = Colour.STRIPES  # show a green circle for a stripes ball
            else:
                colour = Colour.CUE  # show a white circle for any other ball type

            cv2.circle(output, ball.position.tuple(), ball.radius, colour, -1)

        shot = self.__calculate_best_shot(ball_type)  # determine the best shot

        if shot:
            phantom_cue_ball = shot.create_phantom_cue_ball()  # location of cue ball when it hits target ball
            cv2.circle(output, phantom_cue_ball.position.tuple(), phantom_cue_ball.radius, Colour.CUE, 1)

            # show ball trajectories
            cv2.line(output, shot.cue_ball.position.tuple(), phantom_cue_ball.position.tuple(), Colour.CUE)
            cv2.line(output, shot.target_ball.position.tuple(), shot.pocket.position.tuple(), Colour.CUE)
        else:
            cv2.putText(output, "No good shots available", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, Colour.RED)

        cv2.imshow("Output", output)
        cv2.waitKey(0)

    # return the ball object of the cue ball
    def __get_cue_ball(self):
        for ball in self.balls:
            if ball.type == "cue":
                return ball

    # return the shot with the least angle for a ball of type ball_type
    def __calculate_best_shot(self, ball_type):
        possible_shots = self.__calculate_possible_shots(ball_type)

        if len(possible_shots) == 0:
            return None

        return min(possible_shots, key=lambda p: p.angle)

    # calculate all shots with an angle of less than 90 degree for a ball of type ball_type
    def __calculate_possible_shots(self, ball_type):
        cue_ball = self.__get_cue_ball()
        possible_shots = []  # initial array of possible shots

        # loop over balls and pockets and create a shot object for every one
        for target_ball in self.balls:
            if target_ball.type != ball_type:
                continue

            for pocket in self.pockets:
                shot = Shot(cue_ball, target_ball, pocket)

                if shot.angle < 90:  # it's impossible to hit a ball at an angle of greater than 90 degrees
                    possible_shots.append(shot)  # add shot to array

        return possible_shots


class Ball:
    def __init__(self, _type, position):
        assert _type in ["cue", "eight", "solids", "stripes", "phantom_cue"]

        self.type = _type
        self.position = position
        self.radius = 10


class Pocket:
    def __init__(self, position):
        self.position = position
        self.radius = 20


class Shot:
    def __init__(self, cue_ball, target_ball, pocket):
        self.cue_ball = cue_ball
        self.target_ball = target_ball
        self.pocket = pocket

        # use cosine rule to calculate the angle the target ball must deviate to land in a pocket
        self.angle = 180 - Maths.angle_between_points(self.target_ball.position, self.cue_ball.position,
                                                      self.pocket.position)

    # create "phantom" ball where cue ball makes contact with the target ball
    def create_phantom_cue_ball(self):

        # calculate gradient of line between pocket and target ball
        gradient = - (self.pocket.position.y - self.target_ball.position.y) / \
                   (self.pocket.position.x - self.target_ball.position.x)
        angle = math.atan(gradient)  # angle of gradient

        # wizardry
        if self.pocket.position.x > self.target_ball.position.x:
            x = int(self.target_ball.position.x - self.target_ball.radius * 2 * math.cos(angle))
            y = int(self.target_ball.position.y + self.target_ball.radius * 2 * math.sin(angle))
        else:
            x = int(self.target_ball.position.x + self.target_ball.radius * 2 * math.cos(angle))
            y = int(self.target_ball.position.y - self.target_ball.radius * 2 * math.sin(angle))

        return Ball("phantom_cue", Point(x, y))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def tuple(self):
        return self.x, self.y


class Maths:
    # calculate distance between two points
    @staticmethod
    def calculate_distance(position1, position2):
        return math.sqrt((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2)

    @staticmethod
    def radians_to_degrees(radians):
        return radians * 180 / math.pi

    # given points a, b and c, returns angle at c using the cosine rule
    @staticmethod
    def angle_between_points(target, far1, far2):
        a = Maths.calculate_distance(target, far1)
        b = Maths.calculate_distance(target, far2)
        c = Maths.calculate_distance(far1, far2)
        return Maths.radians_to_degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))


class Colour:
    CUE = (255, 255, 255)
    SOLIDS = (255, 0, 0)
    STRIPES = (0, 255, 0)
    POCKET = (0, 0, 255)
    RED = (0, 0, 255)


if __name__ == "__main__":
    # Eventually, we want to actually read in an image at find its features. But this will do for now.

    image = np.zeros((600, 800, 3), np.uint8)  # create a blank black image
    table = Table(image)

    # add pockets to table
    table.pockets.append(Pocket(Point(100, 100)))
    table.pockets.append(Pocket(Point(400, 100)))
    table.pockets.append(Pocket(Point(700, 100)))
    table.pockets.append(Pocket(Point(100, 500)))
    table.pockets.append(Pocket(Point(400, 500)))
    table.pockets.append(Pocket(Point(700, 500)))

    # add balls to table at random location
    table.balls.append(Ball("cue", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("solids", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("solids", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("solids", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("solids", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("stripes", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("stripes", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("stripes", Point(150 + randint(0, 500), 150 + randint(0, 300))))
    table.balls.append(Ball("stripes", Point(150 + randint(0, 500), 150 + randint(0, 300))))

    table.show_best_shot("solids")
