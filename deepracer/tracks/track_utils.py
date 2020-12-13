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

# os tools to list things
from os import listdir
from os.path import isfile, join


class TrackIO:
    """Utility to help load npy files containing the track waypoints.

    Track files are considered to contain a numpy array of threes of waypoint coordinates.
    Each three consists of two coordinates in meters when transfered to real world tracks.
    So effectively each element in waypoints list is a numpy array of six numbers:
    [ center_x, center_y, inner_x, inner_y, outer_x, outer_y ]
    The first and last waypoint are usually the same (closing the loop), but
    it isn't a clear requirement in terms of tracks definiton.
    Some of the waypoints may be identical, this is a state which is accepted by the simulator.
    """

    def __init__(self, base_path="./tracks"):
        """Create the TrackIO instance.

        Arguments:
        base_path - base path pointing to a folder containing the npy files.
                    default value: "./tracks"
        """
        self.base_path = base_path

    def get_tracks(self):
        """Yields tracks in the base_path
        """
        for f in sorted(listdir(self.base_path)):
            if isfile(join(self.base_path, f)) and f.endswith('.npy'):
                yield f

    def get_track_waypoints(self, track_name):
        """Load track waypoints as an array of coordinates

        Truth be told, it will load just about any npy file without checks,
        as long as it has the npy extension.

        Arguments:
        track_name - name of the track to load. Both just the name and name.npy
                     will be accepted

        Returns:
        A list of elements where each element is a numpy array of six float values
        representing coordinates of track points in meters
        """
        if track_name.endswith('.npy'):
            track_name = track_name[:-4]
        return np.load("%s/%s.npy" % (self.base_path, track_name))

    def load_track(self, track_name):
        """Load track waypoints as a Track object

        No validation is being made on the input data, results of running on npy files
        other than track info will provide undetermined results.

        Arguments:
        track_name - name of the track to load. Both just the name and name.npy
                     will be accepted

        Returns:
        A Track instance representing the track
        """
        if track_name.endswith('.npy'):
            track_name = track_name[:-4]

        waypoints = self.get_track_waypoints(track_name)

        print("Loaded %s waypoints" % waypoints.shape[0])

        return Track(track_name, waypoints)


class TrackBreakdown:
    """Track info available to enrich the action breakdown graphs.
    Not essential, but can help find interesting sections to pay attention to.
    """

    def __init__(self, vert_lines, track_segments, segment_x, segment_y,
                 segment_xerr, segment_yerr):
        # vert_lines are indices of waypoints put on the track in squares to mark a section
        self.vert_lines = vert_lines
        # track segments determine location of descriptions on the right graph, formed of tuple
        # (location along the x axis, location along the y axis, description)
        self.track_segments = track_segments

        # marking of a bottom-left pixel of a segment on the right graph
        self.segment_x = segment_x
        self.segment_y = segment_y

        # boundaries of red rectangles on the right graph
        # how many pixels wide before and after the bottom-left pixel x coordinate
        self.segment_xerr = segment_xerr
        # how many pixels tall below and above the bottom-left pixel y coordinate
        self.segment_yerr = segment_yerr


reinvent2018 = TrackBreakdown(
    vert_lines=[10, 25, 32, 33, 40, 45, 50, 53, 61, 67],
    track_segments=[(15, 100, 'hairpin'),
                    (32, 100, 'right'),
                    (42, 100, 'left'),
                    (51, 100, 'left'),
                    (63, 100, 'left')],

    segment_x=np.array([15, 32, 42, 51, 63]),
    segment_y=np.array([0, 0, 0, 0, 0]),

    segment_xerr=np.array([[5, 1, 2, 1, 2], [10, 1, 3, 2, 4]]),
    segment_yerr=np.array([[0, 0, 0, 0, 0], [150, 150, 150, 150, 150]]))

london_loop = TrackBreakdown(
    vert_lines=[0, 15, 17, 30, 33, 45, 75, 105, 120, 132, 150, 180, 190, 210],
    track_segments=[(0, 100, 'long sharp left'),
                    (17, 90, 'mild right'),
                    (33, 80, 'tight left'),
                    (75, 100, 'mild chicane'),
                    (120, 100, 'short sharp left'),
                    (150, 90, 'left'),
                    (190, 100, 'right')],

    segment_x=np.array([0, 17, 33, 75, 120, 150, 190]),
    segment_y=np.array([0, 0, 0, 0, 0, 0, 0]),

    segment_xerr=np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, 13, 12, 30, 12, 30, 20]]),
    segment_yerr=np.array(
        [[0, 0, 0, 0, 0, 0, 0], [150, 150, 150, 150, 150, 150, 150]]))

track_breakdown = {'reinvent2018': reinvent2018, 'london_loop': london_loop}


track_meta = {}
track_meta['Austin'] = 'American Hills Speedway'
track_meta['Singapore'] = 'Asia Pacific Bay Loop'
track_meta['Monaco'] = 'European Seaside Circuit'
track_meta['Aragon'] = 'Stratus Loop'
track_meta['Belille'] = 'Cumulo Turnpike'
track_meta['Albert'] = 'Yun Speedway'
track_meta['July_2020'] = 'Roger Raceway'
track_meta['FS_June2020'] = 'Fumiaki Loop'
track_meta['Spain_track'] = 'Circuit de Barcelona-Catalunya'
track_meta['reInvent2019_track'] = 'The 2019 DeepRacer Championship Cup'
track_meta['reinvent_base'] = 're:Invent 2018'
track_meta['AmericasGeneratedInclStart'] = 'Badaal Track'
track_meta['LGSWide'] = 'SOLA Speedway'
track_meta['Vegas_track'] = 'AWS Summit Raceway'
track_meta['Canada_Training'] = 'Toronto Turnpike Training'
track_meta['Canada_Eval'] = 'Toronto Turnpike Eval'
track_meta['Mexico_track'] = 'Cumulo Carrera Training'
track_meta['Mexico_track_eval'] = 'Cumulo Carrera Eval'
track_meta['China_track'] = 'Shanghai Sudu Training'
track_meta['China_eval_track'] = 'Shanghai Sudu Eval'
track_meta['New_York_Track'] = 'Empire City Training'
track_meta['New_York_Eval_Track'] = 'Empire City Eval'
track_meta['Tokyo_Training_track'] = 'Kumo Torakku Training'
track_meta['Virtual_May19_Train_track'] = 'London Loop Training'
track_meta['Bowtie_track'] = 'Bowtie Track'
track_meta['Oval_track'] = 'Oval Track'
track_meta['reInvent2019_wide'] = 're:Invent 2018 Wide'
track_meta['reInvent2019_wide_mirrored'] = 're:Invent 2018 Wide Mirrored'
track_meta['H_track'] = 'H track'
track_meta['Straight_track'] = 'Straight track'


class Track:
    """Track object represents a track.

    I know, right?

    Fields:
    name - name of the track loaded
    waypoints - input list as received by constructor
    center_line - waypoints along the center of the track with coordinates in meters
    inner_border - waypoints along the inner border of the track with coordinates in meters
    outer_border - waypoints along the outer border of the track with coordinates in meters
    road_poly - a polygon representing the track
    """

    def __init__(self, name, waypoints):
        """Create Track object

        Arguments:
        name - name of the track
        waypoints - values from a npy file for the track
        """
        self.name = name
        self.waypoints = waypoints
        self.center_line = waypoints[:, 0:2]
        self.inner_border = waypoints[:, 2:4]
        self.outer_border = waypoints[:, 4:6]

        l_inner_border = LineString(waypoints[:, 2:4])
        l_outer_border = LineString(waypoints[:, 4:6])
        self.road_poly = Polygon(
            np.vstack((l_outer_border, np.flipud(l_inner_border))))

    def size(self):
        """Track bounding box size tuple
        """
        return (
            np.max(self.outer_border[:, 0]) - np.min(self.outer_border[:, 0]),
            np.max(self.outer_border[:, 1]) - np.min(self.outer_border[:, 1])
        )

    def ratio(self):
        """Track size ratio
        """
        s = self.size()
        return s[1] / s[0]


class GeometryUtils:
    """A set of utilities for use with vectors and points in 2D

    The general idea is to have them work with numpy array representation
    of vectors and points, to simplify working with them.

    Functions work with coordinates provided in numpy arrays as input and
    the results will always be in numpy arrays.

    """

    @staticmethod
    def vector(p1, p2):
        """Convert two points into a vector

        Arguments:
        p1 - first point (either np. array or a list of two floats)
        p2 - second point (either np. array or a list of two floats)

        Returns:
        Vector represented by a numpy array
        """
        return np.subtract(p1, p2)

    @staticmethod
    def get_angle(v1, v2):
        """Calculate an angle between two vectors

        Arguments:
        v1 - first vector (a numpy array)
        v2 - second vector (a numpy array)

        Returns:
        The angle size in degrees
        """

        angle = np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        return np.degrees(angle)

    @staticmethod
    def get_vector_length(v):
        """Calculate the length of a vector, as in magnitude, not amount of coordinates

        Arguments:
        v - a vector (a numpy array)

        Returns:
        Length of the vector (float)
        """

        return np.linalg.norm(v)

    @staticmethod
    def normalize_vector(v):
        """Return a vector scaled to length 1.0

        Arguments:
        v - a vector (a numpy array)

        Returns:
        A normalized vector
        """

        return v / GeometryUtils.get_vector_length(v)

    @staticmethod
    def perpendicular_vector(v):
        """Return a vector perpendicular to one provided

        The output vector is rotated 90 degrees counter-clockwise to input

        Arguments:
        v - a vector (a numpy array)

        Returns:
        A vector perpendicular to input vector
        """

        return np.cross(v, [0, 0, -1])[:2]

    @staticmethod
    def crossing_point_for_two_lines(l1_p1, l1_p2, l2_p1, l2_p2):
        """Returns the point of intersection of the lines

        The lines are passing through l1_p1, l1_p2 and l2_p1, l2_p2

        Result is rounded to three decimal places
        Work by Norbu Tsering https://stackoverflow.com/a/42727584

        Arguments:
        l1_p1 - [x, y] a point on the first line
        l1_p2 - [x, y] another point on the first line
        l2_p1 - [x, y] a point on the second line
        l2_p2 - [x, y] another point on the second line

        Returns:
        Numpy array with coordinates of a point where the two lines cross
        """

        s = np.vstack([l1_p1, l1_p2, l2_p1, l2_p2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return [float('inf'), float('inf')]
        return [np.round(x/z, 3), np.round(y/z, 3)]

    @staticmethod
    def get_a_and_b_for_function(p1, p2):
        """Returns a and b for the function equation y = a*x + b

        Just note it won't work for an equation x = b (when p1_x = p2_x)
        I would normally transpose the coordinates when something like this is needed:
        y' = x
        x' = y
        then a' = 0 and b' = whatever, do the maths and transpose back.

        Arguments:
        p1 - first point on the line
        p2 - second point on the line

        Returns:
        A tuple with a and b for function equation y = a*x + b
        """

        a1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b1 = p2[1] - a1 * p2[0]
        return a1, b1

    @staticmethod
    def get_a_point_on_a_line_closest_to_point(l1_p1, l1_p2, p):
        """Finds a point on a straight line which is closest to a given point

        Arguments:
        l1_p1 - first point on the line
        l1_p2 - second point on the line
        p - point to which we want to search the closest point on the line
        """
        vector = GeometryUtils.perpendicular_vector(GeometryUtils.vector(l1_p1, l1_p2))
        p2 = p + vector
        crossing_point = GeometryUtils.crossing_point_for_two_lines(
            l1_p1, l1_p2, p, p2)
        return crossing_point

    @staticmethod
    def is_point_roughly_on_the_line(lp1, lp2, p, tolerated_angle=5):
        """Tells you if the point is roughly on a line.

        In practice it calculates two angles:
                * <- p
               / \
              /   \
             /a1 a2\
           -*-------*--
            ^lp1    ^lp2

        If the absolute value of both the angles is under tolerated_angle,
        The point is close enough to the line to be considered to be on it.

        Just note that when lp1 and lp2 are far from each other, 5 degrees might
        mean a lot, so the point may be not as close as you may think.

        Arguments:
        lp1 - line point 1
        lp2 - line point 2
        p - point to test
        tolerated_angle - how big of an angle can we tolerate? Default is 5 degrees
        """
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
    """Utility to help when trying to plot a track
    """
    @staticmethod
    def plot_track(to_plot, show=True):
        """Plot waypoints for the track

        Arguments:
        waypoints - waypoints to be plotted or the Track object
        show - whether to plot straight away - you may chose to add more to plot
               default value: True
        """
        import matplotlib.pyplot as plt

        if isinstance(to_plot, Track):
            to_plot = to_plot.waypoints

        for point in to_plot:
            plt.scatter(point[0], point[1], c="blue")
            plt.scatter(point[2], point[3], c="black")
            plt.scatter(point[4], point[5], c="cyan")

        if show:
            plt.show()
