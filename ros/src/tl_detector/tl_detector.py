#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Light
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.bridge = CvBridge()
        self.model_path = rospy.get_param('~model_path')
	# rospy.logwarn('model_path {}'.format(self.model_path))
        
	'''
	Initialize the light classifier first before even we subscribe for
        any topics to make sure when image_cb (see bellow) is called the classifier
	has already initialized
	'''
	self.light_classifier = TLClassifier(self.model_path)
 
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        """
	 Change this topic's name from /traffic_waypoint to /traffic_waypoint_intelli to avoid
         any message type conflict for the same topic name; other student's project may run
         before us and they might use the default message type Int32 for this topic but we 
         are using different message type i.e Light, now with the change of the topic name
         this conflict should not occur.
        """
	self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint_intelli', Light, queue_size=1)

        self.listener = tf.TransformListener()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                    for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

	"""
	    Do not process any images if the car has
	    not started yet and we checked the camera
	    check mark
	"""
	if self.pose == None or self.waypoints == None:
	    return	

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
	light = Light()
 
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
	    if state != TrafficLight.RED and state != TrafficLight.YELLOW:
            	light_wp = -1

            self.last_wp = light_wp	    
	    light.index = light_wp
	    light.state = state 
            self.upcoming_red_light_pub.publish(light)
        else:
	    light.index = self.last_wp
	    light.state = self.last_state 
            self.upcoming_red_light_pub.publish(light)
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	# 
        # derive this state by classification and light coordinate
        #return light.state

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        need_convert = True
        if hasattr(self.camera_image, 'encoding'):
            if self.camera_image.encoding == 'rgb8':
                need_convert = False;

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, self.camera_image.encoding)
        # rospy.logwarn('camera_image.encoding {}, need_convert {}'.format(self.camera_image.encoding, need_convert))
        if need_convert :
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light 
	    (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
	# if we do not know the car position and the waypoint_tree has not initialized yet,
        # no point of finding the closest waypoint
	if(self.pose and self.waypoint_tree):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #find the closest visible traffic light (if one exists)
            #diff = len(self.waypoints.waypoints)
            # checking redlight for 50 waypoints ahead is good for detection
            diff = 50
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]

                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                #Find the closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            #rospy.logwarn("Closest Light state {}".format(state))
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
