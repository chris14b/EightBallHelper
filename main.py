import math
import cv2
import numpy as np
from random import randint
import sys
import transform


class Table:
    def __init__(self, _image, min_ball_radius=8, max_pocket_radius=30, radius_threshold=20, cue_ball_threshold=240,
                 eight_ball_threshold=40, white_pixel_ratio_threshold=0.07, black_pixel_ratio_threshold=0.7,
                 hough_param1=60, hough_param2=30, max_image_width=1200, max_image_height=675):
        self.__image = self.__resize_image(_image, max_image_width, max_image_height)
        self.balls = []
        self.pockets = []
        self.__best_shot = None
        self.__min_ball_radius = min_ball_radius  # minimum radius of a ball in pixels
        self.__max_pocket_radius = max_pocket_radius  # maximum radius of a pocket in pixels
        self.__radius_threshold = radius_threshold  # number between ball radius and pocket radius
        self.__cue_ball_threshold = cue_ball_threshold  # brightness cue ball is above
        self.__eight_ball_threshold = eight_ball_threshold  # brightness eight ball is below
        self.__hough_param1 = hough_param1
        self.__hough_param2 = hough_param2

        # amount of white a ball needs to be considered stripes
        self.__white_pixel_ratio_threshold = white_pixel_ratio_threshold

        # amount of black a ball needs to be considered the eight ball
        self.__black_pixel_ratio_threshold = black_pixel_ratio_threshold

    # show the table, including marked balls, pockets and best shot if available
    def show_best_shot(self):
        output = self.__image

        for pocket in self.pockets:
            # show a red circle for each pocket
            cv2.circle(output, pocket.position.tuple(), pocket.radius, Colour.POCKET, 2)

        for _ball in self.balls:
            if _ball.type == "solids":
                colour = Colour.SOLIDS  # show a blue circle for a solids ball
            elif _ball.type == "stripes":
                colour = Colour.STRIPES  # show a green circle for a stripes ball
            elif _ball.type == "cue":
                colour = Colour.CUE  # show a white circle for any other ball type
            else:
                colour = Colour.EIGHT

            cv2.circle(output, _ball.position.tuple(), _ball.radius, colour, 2)

        if self.__best_shot:
            cv2.circle(output, self.__best_shot.phantom_cue_ball.position.tuple(),
                       self.__best_shot.phantom_cue_ball.radius, Colour.CUE, 1)

            # show ball trajectories
            cv2.line(output, self.__best_shot.cue_ball.position.tuple(),
                     self.__best_shot.phantom_cue_ball.position.tuple(), Colour.CUE)
            cv2.line(output, self.__best_shot.target_ball.position.tuple(),
                     self.__best_shot.pocket.position.tuple(), Colour.CUE)

        else:
            cv2.putText(output, "No good shots available", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, Colour.RED)

        cv2.imshow("Output", output)
        cv2.waitKey(0)

    # return the ball object of the cue ball
    def __get_cue_ball(self):
        for _ball in self.balls:
            if _ball.type == "cue":
                return _ball

        return None

    # return the shot with the least angle for a ball of type ball_type
    def calculate_best_shot(self, ball_type):
        if not self.__get_cue_ball():
            return None

        possible_shots = self.__calculate_possible_shots(ball_type)

        if len(possible_shots) == 0:
            return None

        self.__best_shot = min(possible_shots, key=lambda p: p.angle)

    # calculate all shots with an angle of less than 90 degree for a ball of type ball_type
    def __calculate_possible_shots(self, ball_type):
        cue_ball = self.__get_cue_ball()
        possible_shots = []  # initial array of possible shots

        # loop over balls and pockets and create a shot object for every one
        for target_ball in self.balls:
            if target_ball.type != ball_type:
                continue

            for pocket in self.pockets:
                shot = Shot(cue_ball, target_ball, pocket, self.balls)

                if shot.possible:  # it's impossible to hit a ball at an angle of greater than 90 degrees
                    possible_shots.append(shot)  # add shot to array

        return possible_shots

    # detect circles (ie balls and pockets) in image
    def find_circles_in_image(self):
        gray_img = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.__image, cv2.COLOR_BGR2HSV)
        mask = transform.getTableMask(hsv, transform.findFeltHueAutomatic(hsv, 'hsv'), 'hsv')
        gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, self.__min_ball_radius * 1.5,
                                   param1=self.__hough_param1, param2=self.__hough_param2,
                                   minRadius=self.__min_ball_radius, maxRadius=self.__max_pocket_radius)

        if circles is None:
            return

        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            position = Point(circle[0], circle[1])

            if circle[2] >= self.__radius_threshold:
                self.pockets.append(Pocket(position, radius=circle[2]))
            elif self.__point_on_table(position):
                self.balls.append(Ball(self.__get_ball_type(circle), position, radius=circle[2]))

    # check if a given point is on the table by seeing if it lies within all six pockets
    def __point_on_table(self, point):
        if len(self.pockets) < 6:
            return None

        leftmost = math.inf
        rightmost = -1
        topmost = math.inf
        bottommost = -1

        for pocket in self.pockets:
            leftmost = min(leftmost, pocket.position.x)
            rightmost = max(rightmost, pocket.position.x)
            topmost = min(topmost, pocket.position.y)
            bottommost = max(bottommost, pocket.position.y)

        return leftmost < point.x < rightmost and topmost < point.y < bottommost

    def __get_ball_type(self, circle):
        x, y, radius = circle
        x_min = x - radius
        x_max = x + radius
        y_min = y - radius
        y_max = y + radius

        grey_image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        pixels = []
        num_white_pixels = 0
        num_black_pixels = 0

        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if Maths.distance(Point(x, y), Point(i, j)) <= radius:
                    curr_pixel = grey_image[j, i]
                    pixels.append(curr_pixel)

                    if curr_pixel > self.__cue_ball_threshold:
                        num_white_pixels += 1
                    elif curr_pixel < self.__eight_ball_threshold:
                        num_black_pixels += 1

        num_pixels = len(pixels)
        brightness = sum(pixels) / num_pixels
        white_pixel_ratio = num_white_pixels / num_pixels
        black_pixel_ratio = num_black_pixels / num_pixels

        if brightness > self.__cue_ball_threshold:
            return "cue"
        elif black_pixel_ratio > self.__black_pixel_ratio_threshold:
            return "eight"
        elif white_pixel_ratio < self.__white_pixel_ratio_threshold:
            return "solids"
        else:
            return "stripes"

    # check whether a ball overlaps with other balls on the table
    def __is_overlapping(self, new_ball):
        for curr_ball in self.balls:
            if Maths.distance(new_ball.position, curr_ball.position) <= curr_ball.radius*2:
                return True
        return False

    def add_ball(self, _type):
        new_ball = Ball(_type, Point(150 + randint(0, 500), 150 + randint(0, 300)))

        while self.__is_overlapping(new_ball):
            new_ball.position = Point(150 + randint(0, 500), 150 + randint(0, 300))

        self.balls.append(new_ball)

    # resizes image to fit given dimensions
    @staticmethod
    def __resize_image(_image, max_width, max_height):
        original_height = _image.shape[0]
        original_width = _image.shape[1]

        if original_height > max_height or original_width > max_width:
            aspect_ratio = original_width / original_height

            if aspect_ratio >= max_width / max_height:
                width = max_width
                height = width * _image.shape[0] / _image.shape[1]
                resized_image = cv2.resize(_image, (int(width), int(height)))
            else:
                height = max_height
                width = height * _image.shape[1] / _image.shape[0]
                resized_image = cv2.resize(_image, (int(width), int(height)))
        else:
            return _image

        return resized_image


class Ball:
    def __init__(self, _type, position, radius=10):
        assert _type in ["cue", "eight", "solids", "stripes", "phantom_cue"]

        self.type = _type
        self.position = position
        self.radius = radius


class Pocket:
    def __init__(self, position, radius=20):
        self.position = position
        self.radius = radius


class Shot:
    def __init__(self, cue_ball, target_ball, pocket, all_balls):
        self.cue_ball = cue_ball
        self.target_ball = target_ball
        self.pocket = pocket
        self.all_balls = all_balls
        self.phantom_cue_ball = self.__create_phantom_cue_ball()

        # use cosine rule to calculate the angle the target ball must deviate to land in a pocket
        self.angle = math.pi - Maths.angle_between_points(self.phantom_cue_ball.position, self.cue_ball.position,
                                                          self.pocket.position)

        # a shot is not possible if the ball trajectories are obscured, or if the angle is greater than 90 degrees
        self.possible = self.angle < math.pi / 2 and not self.__shot_obscured()

    # create "phantom" ball where cue ball makes contact with the target ball
    def __create_phantom_cue_ball(self):
        run = self.pocket.position.x - self.target_ball.position.x

        if run != 0:
            # calculate gradient of line between pocket and target ball
            gradient = - (self.pocket.position.y - self.target_ball.position.y) / run
            angle = math.atan(gradient)  # angle of gradient
        else:
            angle = math.pi / 2

        # wizardry
        if self.pocket.position.x > self.target_ball.position.x:
            x = self.target_ball.position.x - self.target_ball.radius * 2 * math.cos(angle)
            y = self.target_ball.position.y + self.target_ball.radius * 2 * math.sin(angle)
        else:
            x = self.target_ball.position.x + self.target_ball.radius * 2 * math.cos(angle)
            y = self.target_ball.position.y - self.target_ball.radius * 2 * math.sin(angle)

        return Ball("phantom_cue", Point(x, y), self.cue_ball.radius)

    # determine whether ball trajectories are obscured by other balls
    def __shot_obscured(self):
        for curr_ball in self.all_balls:
            if curr_ball in [self.cue_ball, self.target_ball]:
                continue

            # wizardry
            if Maths.distance(self.phantom_cue_ball.position, curr_ball.position) < self.phantom_cue_ball.radius + \
                    curr_ball.radius or \
                    Maths.perpendicular_distance(self.cue_ball.position, self.phantom_cue_ball.position,
                                                 curr_ball.position) < self.cue_ball.radius + curr_ball.radius or \
                    Maths.perpendicular_distance(self.target_ball.position, self.pocket.position,
                                                 curr_ball.position) < self.target_ball.radius + curr_ball.radius:
                return True

        return False


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def tuple(self):
        return int(self.x), int(self.y)


class Maths:
    # calculate distance between two points
    @staticmethod
    def distance(position1, position2):
        result = math.sqrt((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2)
        return result

    @staticmethod
    def radians_to_degrees(radians):
        return radians * 180 / math.pi

    # given points a, b and c, returns angle at c using the cosine rule
    @staticmethod
    def angle_between_points(target, far1, far2):
        a = Maths.distance(target, far1)
        b = Maths.distance(target, far2)
        c = Maths.distance(far1, far2)
        return math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    # perpendicular distance between other_point and a line created by points line_point1 and line_point2
    @staticmethod
    def perpendicular_distance(line_point1, line_point2, other_point):
        angle = Maths.angle_between_points(line_point1, line_point2, other_point)
        distance = Maths.distance(line_point1, other_point)

        if angle < math.pi / 2 and math.fabs(distance * math.cos(angle)) <= Maths.distance(line_point1, line_point2):
            return math.fabs(distance * math.sin(angle))
        else:
            return math.inf


class Colour:
    CUE = (255, 255, 255)
    SOLIDS = (0, 0, 255)
    STRIPES = (0, 255, 255)
    POCKET = (10, 36, 72)
    TABLE = (10, 108, 3)
    EIGHT = (0, 0, 0)
    RED = (0, 0, 255)


if __name__ == "__main__":
    # First command line argument will be the file name of image. If none is supplied, generate random table
    if len(sys.argv) == 1:
        while True:
            image = np.zeros((600, 800, 3), np.uint8)  # create a blank black image
            image[:] = Colour.TABLE
            table = Table(image)

            # add pockets to table
            table.pockets.append(Pocket(Point(100, 100)))
            table.pockets.append(Pocket(Point(400, 100)))
            table.pockets.append(Pocket(Point(700, 100)))
            table.pockets.append(Pocket(Point(100, 500)))
            table.pockets.append(Pocket(Point(400, 500)))
            table.pockets.append(Pocket(Point(700, 500)))

            # add balls to table at random location
            table.add_ball("cue")

            for ball in range(7):    
                table.add_ball("solids")
                table.add_ball("stripes")
                
            table.calculate_best_shot("solids")
            table.show_best_shot()
    else:
        image = cv2.imread(sys.argv[1])

        if sys.argv[1] == "table_1.jpg":
            table = Table(image)
        elif sys.argv[1] == "table_2.jpg":
            table = Table(image, min_ball_radius=11, max_pocket_radius=30, cue_ball_threshold=220,
                          eight_ball_threshold=40, white_pixel_ratio_threshold=0.2, black_pixel_ratio_threshold=0.31,
                          hough_param1=60, hough_param2=27)
        elif sys.argv[1] == "8BallSampleFrame1.png":
            table = Table(image, min_ball_radius=10,
                          max_pocket_radius=30, radius_threshold=15, hough_param1=80, hough_param2=15)
        else:
            table = Table(image)

        table.find_circles_in_image()
        table.calculate_best_shot("solids")
        print("Displaying image...")
        table.show_best_shot()
