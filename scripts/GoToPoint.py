#!/usr/bin/env python

# COM760 Group 30 - Collapsed School Rescue Robot
# GoToPoint.py - Moves robot from current position to target point
# Part of Bug2 navigation algorithm

import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf import transformations
from com760cw2_com760group30.srv import (
    MineRescueSetBugStatus,
    MineRescueSetBugStatusResponse)

class GoToPoint:

    def __init__(self):
        rospy.init_node('go_to_point')

        self.active  = False
        self.state   = 0
        self.state_labels = {
            0: 'Fix heading',
            1: 'Go straight',
            2: 'Goal reached'
        }

        self.position = Point()
        self.yaw      = 0.0

        self.desired_position   = Point()
        self.desired_position.x = 8.0
        self.desired_position.y = 0.0

        self.linear_speed  = 0.5
        self.angular_speed = 0.5
        self.yaw_threshold  = math.pi / 90
        self.dist_threshold = 0.25

        self.pub_vel = rospy.Publisher(
            '/com760group30Bot/cmd_vel', Twist, queue_size=1)

        self.sub_odom = rospy.Subscriber(
            '/com760group30Bot/odom', Odometry, self.callback_odom)

        self.srv = rospy.Service(
            'go_to_point_switch',
            MineRescueSetBugStatus,
            self.handle_switch)

        rospy.loginfo('[GoToPoint] Ready and waiting...')

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.active:
                rate.sleep()
                continue
            if self.state == 0:
                self.fix_heading(self.desired_position)
            elif self.state == 1:
                self.go_straight(self.desired_position)
            elif self.state == 2:
                self.done()
            else:
                rospy.logerr('[GoToPoint] Unknown state!')
            rate.sleep()

    def handle_switch(self, req):
        self.active = req.flag
        if req.flag:
            if req.speed > 0:
                self.linear_speed = req.speed
            self.desired_position.x = req.goal_x
            self.desired_position.y = req.goal_y
            self.state = 0
            rospy.loginfo(
                '[GoToPoint] Activated. Goal: (%.2f, %.2f)',
                req.goal_x, req.goal_y)
            return MineRescueSetBugStatusResponse(
                success=True,
                message='GoToPoint activated')
        else:
            self.stop_robot()
            return MineRescueSetBugStatusResponse(
                success=True,
                message='GoToPoint deactivated')

    def callback_odom(self, msg):
        self.position = msg.pose.pose.position
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler    = transformations.euler_from_quaternion(quaternion)
        self.yaw = euler[2]

    def fix_heading(self, target):
        desired_yaw = math.atan2(
            target.y - self.position.y,
            target.x - self.position.x)
        yaw_error = self.normalise_angle(desired_yaw - self.yaw)
        msg = Twist()
        if math.fabs(yaw_error) > self.yaw_threshold:
            msg.angular.z = (self.angular_speed
                if yaw_error > 0 else -self.angular_speed)
        else:
            msg.angular.z = 0.0
            self.change_state(1)
        self.pub_vel.publish(msg)

    def go_straight(self, target):
        dist = self.distance_to(target)
        if dist > self.dist_threshold:
            desired_yaw = math.atan2(
                target.y - self.position.y,
                target.x - self.position.x)
            yaw_error = self.normalise_angle(desired_yaw - self.yaw)
            msg = Twist()
            msg.linear.x  = self.linear_speed
            msg.angular.z = 0.3 * yaw_error
            self.pub_vel.publish(msg)
        else:
            self.change_state(2)

    def done(self):
        self.stop_robot()
        rospy.loginfo('[GoToPoint] Goal reached!')
        self.active = False

    def change_state(self, new_state):
        if self.state != new_state:
            rospy.loginfo(
                '[GoToPoint] State: %s -> %s',
                self.state_labels[self.state],
                self.state_labels[new_state])
            self.state = new_state

    def stop_robot(self):
        self.pub_vel.publish(Twist())

    def distance_to(self, target):
        return math.sqrt(
            (target.x - self.position.x)**2 +
            (target.y - self.position.y)**2)

    @staticmethod
    def normalise_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

if __name__ == '__main__':
    try:
        GoToPoint()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass