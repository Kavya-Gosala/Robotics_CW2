#!/usr/bin/env python

# COM760 Group 30 - Collapsed School Rescue Robot
# SurvivorDetector.py - Detects survivors in collapsed school building
# Publishes SurvivorDetected message when robot is close to a survivor

import rospy
import math
from nav_msgs.msg import Odometry
from com760cw2_com760group30.msg import SurvivorDetected

class SurvivorDetector:

    def __init__(self):
        rospy.init_node('survivor_detector')

        # 3 survivor locations in the collapsed school
        # Child 1 at (-6, 3) - trapped in classroom
        # Teacher at (0, -3) - trapped in corridor
        # Child 2 at (5, 3) - trapped near east corridor
        self.survivors = [
            {'id': 1, 'x': -8.0, 'y':  3.0,
             'found': False, 'type': 'Child'},
            {'id': 2, 'x':  0.0, 'y': -3.0,
             'found': False, 'type': 'Teacher'},
            {'id': 3, 'x':  7.0, 'y':  3.0,
             'found': False, 'type': 'Child'},
        ]

        # Detection range in metres
        self.detection_range = 1.5

        self.position_x    = 0.0
        self.position_y    = 0.0
        self.total_found   = 0

        # Publisher for survivor detection alerts
        self.pub_survivor = rospy.Publisher(
            '/com760group30Bot/survivor_detected',
            SurvivorDetected, queue_size=10)

        # Subscriber for robot position
        self.sub_odom = rospy.Subscriber(
            '/com760group30Bot/odom',
            Odometry, self.callback_odom)

        rospy.loginfo('='*50)
        rospy.loginfo('[SchoolRescue] Robot deployed to collapsed school!')
        rospy.loginfo('[SchoolRescue] Searching for 3 survivors...')
        rospy.loginfo('[SchoolRescue] Child 1   at (-6.0,  3.0)')
        rospy.loginfo('[SchoolRescue] Teacher   at ( 0.0, -3.0)')
        rospy.loginfo('[SchoolRescue] Child 2   at ( 5.0,  3.0)')
        rospy.loginfo('='*50)

        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.check_survivors()
            rate.sleep()

    def callback_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y

    def check_survivors(self):
        for survivor in self.survivors:
            if survivor['found']:
                continue

            dist = math.sqrt(
                (self.position_x - survivor['x'])**2 +
                (self.position_y - survivor['y'])**2)

            if dist < self.detection_range:
                survivor['found'] = True
                self.total_found += 1

                # Publish custom SurvivorDetected message
                msg = SurvivorDetected()
                msg.survivor_id  = survivor['id']
                msg.position_x   = survivor['x']
                msg.position_y   = survivor['y']
                msg.distance     = dist
                msg.status       = 'ALIVE'
                msg.timestamp    = str(rospy.get_time())
                self.pub_survivor.publish(msg)

                rospy.logwarn('='*50)
                rospy.logwarn(
                    '*** SCHOOL SURVIVOR FOUND! ***')
                rospy.logwarn(
                    '*** Type: %s ***', survivor['type'])
                rospy.logwarn(
                    '*** ID: %d Location: (%.1f, %.1f) ***',
                    survivor['id'],
                    survivor['x'],
                    survivor['y'])
                rospy.logwarn(
                    '*** Distance: %.2f metres ***', dist)
                rospy.logwarn(
                    '*** Total found: %d/3 ***',
                    self.total_found)
                rospy.logwarn('='*50)

                if self.total_found == 3:
                    rospy.logwarn('='*50)
                    rospy.logwarn(
                        '*** ALL SCHOOL SURVIVORS FOUND! ***')
                    rospy.logwarn(
                        '*** Robot returning to emergency base! ***')
                    rospy.logwarn('='*50)

if __name__ == '__main__':
    try:
        SurvivorDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass