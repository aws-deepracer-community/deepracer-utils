"""
Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Copyright 2019-2020 AWS DeepRacer Community. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

# Shapely Library
from shapely.geometry import Polygon
from shapely.geometry.polygon import LineString


class TrackLoader:
    def __init__(self, base_path="./tracks"):
        self.base_path = base_path

    def get_track_waypoints(self, track_name):
        if track_name.endswith('.npy'):
            track_name = track_name[:-4]
        return np.load("%s/%s.npy" % (self.base_path, track_name))

    def load_track(self, track_name):
        if track_name.endswith('.npy'):
            track_name = track_name[:-4]

        waypoints = self.get_track_waypoints(track_name)

        print("Loaded %s waypoints" % waypoints.shape[0])

        return TrackInfo(waypoints)


class TrackInfo:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.center_line = waypoints[:, 0:2] * 100
        self.inner_border = waypoints[:, 2:4] * 100
        self.outer_border = waypoints[:, 4:6] * 100

        l_inner_border = LineString(waypoints[:, 2:4])
        l_outer_border = LineString(waypoints[:, 4:6])
        self.road_poly = Polygon(
            np.vstack((l_outer_border, np.flipud(l_inner_border))))


class GeometryUtils:
    @staticmethod
    def vector(p1, p2):
        return np.subtract(p1, p2)

    @staticmethod
    def get_angle(v1, v2):
        angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        return np.degrees(angle)

    @staticmethod
    def get_vector_length(v):
        return np.linalg.norm(v)

    @staticmethod
    def normalize_vector(v):
        return v / GeometryUtils.get_vector_length(v)

    @staticmethod
    def perpendicular_vector(v):
        return np.cross(v, [0, 0, -1])[:2]

    @staticmethod
    def perpendicular_normalized_vector_to_straight_line(v):
        p_v = GeometryUtils.perpendicular_vector(v)

        return GeometryUtils.normalize_vector(p_v)

    @staticmethod
    def crossing_point_for_two_lines(l1_p1, l1_p2, l2_p1, l2_p2):
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        Work by Norbu Tsering https://stackoverflow.com/a/42727584

        l1_p1: [x, y] a point on the first line
        l1_p2: [x, y] another point on the first line
        l2_p1: [x, y] a point on the second line
        l2_p2: [x, y] another point on the second line
        """
        s = np.vstack([l1_p1, l1_p2, l2_p1, l2_p2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    @staticmethod
    def get_a_and_b_for_line(p1, p2):
        a1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b1 = p2[1] - a1 * p2[0]
        return a1, b1

    @staticmethod
    def get_a_point_on_a_line_closest_to_point(l1_p1, l1_p2, p):
        vector = GeometryUtils.perpendicular_normalized_vector_to_straight_line(
            GeometryUtils.vector(l1_p1, l1_p2))
        p2 = p + vector
        crossing_point = GeometryUtils.crossing_point_for_two_lines(
            l1_p1, l1_p2, p, p2)
        return crossing_point

    @staticmethod
    def is_point_on_the_line(lp1, lp2, p):
        a1 = GeometryUtils.get_angle(
            GeometryUtils.vector(lp1, lp2),
            GeometryUtils.vector(lp1, p)
        )

        a2 = GeometryUtils.get_angle(
            GeometryUtils.vector(lp2, p),
            GeometryUtils.vector(lp2, p)
        )
        return a1 < 5 and a2 < 5


class TrackPlotter:
    @staticmethod
    def plot_trackpoints(trackpoints, show=True):
        import matplotlib.pyplot as plt
        for point in trackpoints:
            plt.scatter(point[0], point[1], c="blue")
            plt.scatter(point[2], point[3], c="black")
            plt.scatter(point[4], point[5], c="cyan")
        if show:
            plt.show()
