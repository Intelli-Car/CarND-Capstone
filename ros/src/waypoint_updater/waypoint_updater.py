#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, Light, TrafficLight
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

#LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

'''
Number of waypoints we will publish, let keep it small to drive smoothly until
we find a reason to make it longer i.e. smooth braking ?
'''
# Number of waypoints we will publish, let keep it small to drive smoothly until
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish, let keep it small to drive smoothly until
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Current positions
        self.pose = None

        # Way points for the track
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        # Stop line for a traffic light
        self.stopline_wp_idx = -1
        self.stopline_wp_state = TrafficLight.UNKNOWN

        # Number of waypoints that we will be looking ahead
        self.lookahead_wps = 0

        # Max velocity the car is allowed to drive.
        # ros param in kilometer per hour converted to mile per second
        self.max_velocity = 0


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        """
	 Change this topic's name from /traffic_waypoint to /traffic_waypoint_intelli to avoid
         any message type conflict for the same topic name; other student's project may run
         before us and they might use the default message type Int32 for this topic but we 
         are using different message type i.e Light, now with the change of the topic name
         this conflict should not occur.
        """
	rospy.Subscriber('/traffic_waypoint_intelli', Light, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        #Add other member variables you need below

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.waypoint_tree:

                #Get closest waypoint
                self.publish_waypoints()
                rate.sleep()

    def get_closest_waypoint_idx(self):

        # car's current position
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # based on current position find the index of closest way points
        closest_idx = self.waypoint_tree.query([x, y],1)[1]

        # Check if the closest point found above is ahead or behind the car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            # Closest waypoint is behind, take the next one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self):
        ''' publish the final waypoints while considering traffic light ahead and
            if there is a red light, set the velocity on the stop line for that
            red light to zero else set the velocity to Maximum allowed velocity.
        '''
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):

        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            
	    #If no traffic light close by within the lookahead limit
            # just check the max speed and let the car move on
            lane.waypoints = self.set_waypoints_velocities(base_waypoints, -1)
        else:
            lane.waypoints = self.set_waypoints_velocities(base_waypoints, closest_idx)

        return lane

    def set_waypoints_velocities(self, waypoints, closest_idx):
        '''
        if closest_idx == -1, set the velocity to max allowed velocity
        for the given waypoints, else set the velocity to some lower
        number or zere while not exceeding the jerk limit.
        '''

        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            if closest_idx == -1:
                
		# take the minimum of current velocity and max allowed. If we are at the
		# end of the waypoints the current velocity would gradually decreases as
                # has done at waypoint_loader and the car will stop, in any location
		# other than a traffic light follow  the max speed limit.
		p.twist.twist.linear.x = min(wp.twist.twist.linear.x, self.max_velocity)
            else:
                # Three waypoints back from stop line so front of car stops at stop line
                stop_idx = max(self.stopline_wp_idx - closest_idx -3, 0)
                vel = 0.0
		dist = self.distance(waypoints, i, stop_idx)

		if self.stopline_wp_state == TrafficLight.YELLOW:
			
			# For yellow light, slow down a little bit 
			vel = math.sqrt(MAX_DECEL * dist)
			#rospy.logwarn("Yellow light, slowing down")	
			
		if self.stopline_wp_state == TrafficLight.RED:
 			
			if i > stop_idx:
			
				# Stop Immediately as the car passed the stop line
				#rospy.logwarn("Car already passed the stop line, STOP!")	
				vel = 0.0
			else:
                		#slow gradually while not exceeding the jerk
                		vel = math.sqrt(2* MAX_DECEL * dist)
                		if vel < 1.0:
                    			vel = 0.0

                p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

            temp.append(p)

        return temp

    def pose_cb(self, msg):
        ''' Callback for receving current position of the car '''
        self.pose = msg

    def kmph2mps(self, velocity_kmph):
        return (velocity_kmph * 1000.) / (60. * 60.)


    def waypoints_cb(self, waypoints):
        ''' Callback for receiving the way points for the track.
            Also Convert this waypoints into a 2D list i.e.
            [[x1,y1], [x2,y2], [x3,y3]...] and saved it to waypoints_2d.

            Save a K-D tree representation of this waypoints for faster
            search of closest way point in O(logn) time.
            '''

	max_vel = self.kmph2mps(rospy.get_param('/waypoint_loader/velocity'))
        if self.max_velocity != max_vel:
	    self.max_velocity = max_vel
            rospy.logwarn("Max velocity is set to {} meter per second".format(self.max_velocity))

	self.base_lane = waypoints
        self.base_lane_size = np.shape(waypoints.waypoints)[0]
        self.lookahead_wps = min(LOOKAHEAD_WPS, self.base_lane_size//2)

        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                                for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        ''' Callback for /traffic_waypoint message.'''

        # Populate the next traffic light's stop line index
	self.stopline_wp_idx = msg.index
	
	# Save the next traffic light's color
	self.stopline_wp_state = msg.state
			
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        '''
        return the maximum velocity for a given waypoint
        that is are set by the  waypoint_loader
        '''
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        ''' 
	Calculate the Euclidean distance between 2 waypoints, wp1 & wp2 are first 
	and second waypoint index accordingly
	'''
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
