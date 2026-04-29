#!/usr/bin/env python3

# COM760 Group 30 - Collapsed School Rescue Robot
# Bug2.py - Master coordinator implementing Bug2 algorithm
# Mission: Child 1 (-6,3) -> Teacher (0,-3) -> Child 2 (5,3) -> Emergency Base (11,0)
# Auto-starts navigation once sensors are ready.

import rospy
import math
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf import transformations
from com760cw2_com760group30.srv import (
    MineRescueSetBugStatus,
    MineRescueSetBugStatusRequest,
    MineRescueSetBugStatusResponse)
from com760cw2_com760group30.msg import SurvivorDetected

class Bug2:

    def __init__(self):
        rospy.init_node('bug2_coordinator')

        # Navigation states
        # 0 = Standing by   — waiting for sensors, then auto-starts
        # 1 = GoToPoint     — driving straight toward goal
        # 2 = FollowWall    — circumnavigating an obstacle
        # 3 = Waypoint reached — brief pause before next goal
        self.nav_state = 0
        self.nav_labels = {
            0: 'Standing by',
            1: 'GoToPoint',
            2: 'FollowWall',
            3: 'Waypoint reached'
        }

        # Robot pose
        self.position = Point()
        self.yaw      = 0.0

        # Mission waypoints — loaded from ROS parameter server.
        # Edit coordinates in school_rescue_bug2.launch, not here.
        self.waypoints = [
            (rospy.get_param('survivor_1_x', -6.0),
             rospy.get_param('survivor_1_y',  3.0)),
            (rospy.get_param('survivor_2_x',  0.0),
             rospy.get_param('survivor_2_y', -3.0)),
            (rospy.get_param('survivor_3_x',  5.0),
             rospy.get_param('survivor_3_y',  3.0)),
            (rospy.get_param('ambulance_x',  11.0),
             rospy.get_param('ambulance_y',   0.0)),
        ]
        self.waypoint_labels = [
            'Survivor 1',
            'Survivor 2',
            'Survivor 3',
            'Ambulance — Mission Complete',
        ]
        # Survivor waypoints (all non-last waypoints are survivors)
        self.survivor_waypoints = {0, 1, 2}
        self.survivor_detected  = [False, False, False]   # per-waypoint latch
        self.survivor_range     = 2.0   # metres — detect even during wall-follow
        self.current_waypoint = 0

        # Current navigation goal
        self.goal   = Point()
        self.goal.x = self.waypoints[0][0]
        self.goal.y = self.waypoints[0][1]

        # M-line parameters (Bug2 algorithm)
        self.start_position   = Point()
        self.m_line_slope     = 0.0
        self.m_line_intercept = 0.0

        # Obstacle hit-point tracking
        self.obstacle_hit_point = Point()
        self.obstacle_hit_dist  = float('inf')
        self.wall_follow_start  = 0.0   # timestamp when FollowWall began

        # Navigation parameters
        self.speed          = 1.0   # m/s — faster now that targets are known
        self.direction      = 'left'
        self.wall_dir_index = 0   # incremented each recovery; alternates left/right

        # Detection thresholds — tuned for fast direct navigation to known coordinates
        self.obstacle_threshold   = 0.45  # start avoiding obstacles a bit earlier
        self.m_line_threshold     = 0.50  # more lenient: return to direct path sooner
        self.goal_threshold       = 1.2   # consider waypoint reached within 1.2 m
        self.min_wall_follow_secs = 3.0   # only 3 s before checking M-line again
        self.m_line_progress_buf  = 0.20  # just 0.2 m closer is enough to cut back
        self.max_wall_follow_secs = 20.0  # force direct path after 20 s max

        # Loop / stuck detection
        self.loop_detect_radius   = 1.0   # if robot returns within 1 m of hit-point → loop
        self.stuck_check_interval = 8.0   # seconds between progress samples
        self.stuck_min_progress   = 0.3   # robot must move ≥ 0.3 m per interval
        self.stuck_check_time     = 0.0
        self.stuck_check_pos      = Point()

        # Laser front distance
        self.laser_front = float('inf')

        # Sensor-ready flags (auto-start fires once both are True)
        self.got_odom    = False
        self.got_laser   = False
        self.mission_done = False   # prevents restart after mission complete

        # ---- ROS Publishers / Subscribers ----
        self.pub_vel = rospy.Publisher(
            '/com760group30Bot/cmd_vel', Twist, queue_size=1)

        self.pub_survivor = rospy.Publisher(
            '/com760group30Bot/survivor_detected',
            SurvivorDetected, queue_size=10)

        self.sub_odom = rospy.Subscriber(
            '/com760group30Bot/odom', Odometry, self.callback_odom)

        self.sub_laser = rospy.Subscriber(
            '/com760group30Bot/laser/scan', LaserScan, self.callback_laser)

        self.sub_survivor = rospy.Subscriber(
            '/com760group30Bot/survivor_detected',
            SurvivorDetected, self.callback_survivor)

        # ---- Service: external homing override ----
        self.srv_homing = rospy.Service(
            'mine_rescue_homing',
            MineRescueSetBugStatus,
            self.handle_homing_signal)

        # ---- Wait for sub-behaviour services ----
        rospy.loginfo('[Bug2] Waiting for navigation services...')
        rospy.wait_for_service('go_to_point_switch')
        rospy.wait_for_service('wall_follower_switch')

        self.client_gtp = rospy.ServiceProxy(
            'go_to_point_switch', MineRescueSetBugStatus)
        self.client_fw = rospy.ServiceProxy(
            'wall_follower_switch', MineRescueSetBugStatus)

        rospy.loginfo('=' * 55)
        rospy.loginfo('[Bug2] SYSTEM READY — Mission waypoints:')
        for i, (wp, lbl) in enumerate(
                zip(self.waypoints, self.waypoint_labels)):
            rospy.loginfo('  %d. %s -> (%.1f, %.1f)', i + 1, lbl, wp[0], wp[1])
        rospy.loginfo('[Bug2] Auto-starting once sensors are live...')
        rospy.loginfo('=' * 55)

        # ---- Main control loop ----
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.nav_state == 0:
                self.stand_by()
            elif self.nav_state == 1:
                self.bug2_go_to_point()
            elif self.nav_state == 2:
                self.bug2_follow_wall()
            elif self.nav_state == 3:
                self.waypoint_reached_behaviour()
            self.check_proximity_detections()
            self.check_stuck()
            rate.sleep()

    # ------------------------------------------------------------------
    # State 0 — Stand by: auto-start once odometry and laser are ready
    # ------------------------------------------------------------------

    def stand_by(self):
        if self.mission_done:
            return   # mission complete — don't restart
        if self.got_odom and self.got_laser:
            rospy.loginfo('[Bug2] Sensors live. Starting mission.')
            self.start_go_to_point()

    # ------------------------------------------------------------------
    # State 1 — GoToPoint: drive directly toward goal; bail on obstacle
    # ------------------------------------------------------------------

    def bug2_go_to_point(self):
        dist = self.distance_to_goal()

        if dist < self.goal_threshold:
            self.deactivate_go_to_point()
            self.change_state(3)
            return

        if self.laser_front < self.obstacle_threshold:
            # Record where we hit the obstacle
            self.obstacle_hit_point.x = self.position.x
            self.obstacle_hit_point.y = self.position.y
            self.obstacle_hit_dist    = dist
            rospy.loginfo(
                '[Bug2] Obstacle at %.2f m. Switching to FollowWall.', dist)
            self.deactivate_go_to_point()
            self.start_follow_wall()

    # ------------------------------------------------------------------
    # State 2 — FollowWall: hug obstacle; switch back when on M-line
    # ------------------------------------------------------------------

    def bug2_follow_wall(self):
        dist = self.distance_to_goal()

        if dist < self.goal_threshold:
            self.deactivate_follow_wall()
            self.change_state(3)
            return

        elapsed = rospy.get_time() - self.wall_follow_start

        # Only check M-line return after minimum wall-follow time
        if elapsed < self.min_wall_follow_secs:
            return

        # Loop detection: robot circled back close to where it hit the obstacle
        # with no net progress toward the goal → it's looping, recover immediately.
        hit_return = math.sqrt(
            (self.position.x - self.obstacle_hit_point.x) ** 2 +
            (self.position.y - self.obstacle_hit_point.y) ** 2)
        if (elapsed > self.min_wall_follow_secs * 2 and
                hit_return < self.loop_detect_radius and
                dist >= self.obstacle_hit_dist - self.m_line_progress_buf):
            rospy.logwarn(
                '[Bug2] Loop detected: back near hit-point (%.2f m) with no goal progress.'
                ' Recovering.', hit_return)
            self.deactivate_follow_wall()
            self._recover_from_stuck()
            return

        # Escape trap: if wall-following too long without M-line progress,
        # force GoToPoint from the current position so the robot doesn't loop forever.
        if elapsed > self.max_wall_follow_secs:
            rospy.logwarn(
                '[Bug2] Wall-follow timeout (%.0f s). Recovering.', elapsed)
            self.deactivate_follow_wall()
            self._recover_from_stuck()
            return

        # Return to GoToPoint when back on M-line AND significantly closer
        if (self.on_m_line() and
                dist < self.obstacle_hit_dist - self.m_line_progress_buf):
            rospy.loginfo(
                '[Bug2] On M-line at dist=%.2f (hit=%.2f). Back to GoToPoint.',
                dist, self.obstacle_hit_dist)
            self.deactivate_follow_wall()
            # Recompute M-line from new position
            self.start_position.x = self.position.x
            self.start_position.y = self.position.y
            self.compute_m_line()
            self.obstacle_hit_dist = float('inf')
            self.start_go_to_point()

    # ------------------------------------------------------------------
    # State 3 — Waypoint reached: advance to next goal
    # ------------------------------------------------------------------

    def waypoint_reached_behaviour(self):
        lbl = self.waypoint_labels[self.current_waypoint]
        is_last     = self.current_waypoint == len(self.waypoints) - 1
        is_survivor = self.current_waypoint in self.survivor_waypoints

        self.stop_robot()

        if is_last:
            # ---- Emergency base reached ----
            rospy.logwarn('=' * 55)
            rospy.logwarn('[Bug2] *** EMERGENCY BASE REACHED ***')
            rospy.logwarn('[Bug2] *** MISSION COMPLETE ***')
            rospy.logwarn('[Bug2] *** All survivors located and reported ***')
            rospy.logwarn('=' * 55)
            self.mission_done = True
            self.change_state(0)   # stand by — mission done (won't restart)
            return
        elif is_survivor:
            # ---- Survivor found: stop, scan, publish, signal ----
            rospy.logwarn('=' * 55)
            rospy.logwarn('[Bug2] *** SURVIVOR LOCATION REACHED ***')
            rospy.logwarn('[Bug2] *** %s ***', lbl)
            rospy.logwarn('[Bug2] *** Position: (%.1f, %.1f) ***',
                          self.goal.x, self.goal.y)
            rospy.logwarn('[Bug2] *** Scanning for signs of life... ***')
            rospy.logwarn('=' * 55)

            # Publish detection (skip if proximity check already fired for this one)
            if not self.survivor_detected[self.current_waypoint]:
                self.survivor_detected[self.current_waypoint] = True
                det_msg              = SurvivorDetected()
                det_msg.survivor_id  = self.current_waypoint + 1
                det_msg.position_x   = self.goal.x
                det_msg.position_y   = self.goal.y
                det_msg.distance     = self.distance_to_goal()
                det_msg.status       = 'ALIVE'
                det_msg.timestamp    = str(rospy.get_time())
                self.pub_survivor.publish(det_msg)

            rospy.sleep(3.0)   # pause at survivor location
            rospy.logwarn('[Bug2] *** SIGNAL SENT ***')
            rospy.logwarn('=' * 55)

            # If all survivors were found during the sleep, _redirect_to_emergency_base
            # already set current_waypoint to the last index — don't override it.
            if self.current_waypoint >= len(self.waypoints) - 1:
                return
        else:
            # ---- Transit waypoint: no pause, continue immediately ----
            rospy.loginfo('[Bug2] Transit waypoint %s reached.', lbl)

        self.current_waypoint += 1
        next_wp = self.waypoints[self.current_waypoint]
        self.goal.x = next_wp[0]
        self.goal.y = next_wp[1]
        rospy.loginfo('[Bug2] Next target: %s (%.1f, %.1f)',
                      self.waypoint_labels[self.current_waypoint],
                      next_wp[0], next_wp[1])

        self.start_position.x = self.position.x
        self.start_position.y = self.position.y
        self.compute_m_line()
        self.obstacle_hit_dist = float('inf')
        self.start_go_to_point()

    # ------------------------------------------------------------------
    # Continuous proximity survivor detection (runs every loop tick)
    # ------------------------------------------------------------------

    def check_proximity_detections(self):
        """Publish SurvivorDetected whenever the robot is within survivor_range
        of any survivor waypoint, regardless of current nav state.  Uses a
        per-survivor latch so each is published exactly once.
        When all 3 are found, immediately redirect to the emergency base."""
        if not self.got_odom or self.mission_done:
            return
        for idx in self.survivor_waypoints:
            if self.survivor_detected[idx]:
                continue
            wp_x, wp_y = self.waypoints[idx]
            dist = math.sqrt(
                (self.position.x - wp_x) ** 2 +
                (self.position.y - wp_y) ** 2)
            if dist < self.survivor_range:
                self.survivor_detected[idx] = True
                det_msg             = SurvivorDetected()
                det_msg.survivor_id = idx + 1
                det_msg.position_x  = wp_x
                det_msg.position_y  = wp_y
                det_msg.distance    = dist
                det_msg.status      = 'ALIVE'
                det_msg.timestamp   = str(rospy.get_time())
                self.pub_survivor.publish(det_msg)
                rospy.logwarn('[Bug2] *** PROXIMITY DETECTION: survivor %d at'
                              ' (%.1f, %.1f), dist=%.2f m ***',
                              idx + 1, wp_x, wp_y, dist)

        # All 3 found — cut straight to the emergency base from here
        if (all(self.survivor_detected) and
                not self.mission_done and
                self.current_waypoint < len(self.waypoints) - 1):
            self._redirect_to_emergency_base()

    def _redirect_to_emergency_base(self):
        """Abort current navigation and head directly to the emergency base.
        Called from the main loop thread so there is no concurrent-write risk."""
        rospy.logwarn('=' * 55)
        rospy.logwarn('[Bug2] *** ALL SURVIVORS FOUND ***')
        rospy.logwarn('[Bug2] *** SHORTEST ROUTE TO EMERGENCY BASE ***')
        rospy.logwarn('=' * 55)
        self.current_waypoint  = len(self.waypoints) - 1
        self.goal.x            = self.waypoints[-1][0]
        self.goal.y            = self.waypoints[-1][1]
        self.obstacle_hit_dist = float('inf')   # reset M-line bias
        if self.nav_state == 2:
            self.deactivate_follow_wall()
        elif self.nav_state == 1:
            self.deactivate_go_to_point()
        self.start_go_to_point()

    # ------------------------------------------------------------------
    # Stuck / loop recovery
    # ------------------------------------------------------------------

    def check_stuck(self):
        """Position-progress check: if the robot hasn't moved enough in
        stuck_check_interval seconds while actively navigating, trigger
        recovery.  Catches tight-corner spinning that loop detection misses."""
        if self.mission_done or not self.got_odom:
            return
        now = rospy.get_time()
        if self.nav_state not in (1, 2):
            # Not actively navigating — keep the reference position fresh.
            self.stuck_check_time    = now
            self.stuck_check_pos.x   = self.position.x
            self.stuck_check_pos.y   = self.position.y
            return
        if now - self.stuck_check_time < self.stuck_check_interval:
            return
        moved = math.sqrt(
            (self.position.x - self.stuck_check_pos.x) ** 2 +
            (self.position.y - self.stuck_check_pos.y) ** 2)
        self.stuck_check_pos.x = self.position.x
        self.stuck_check_pos.y = self.position.y
        self.stuck_check_time  = now
        if moved < self.stuck_min_progress:
            rospy.logwarn(
                '[Bug2] Stuck detected: moved only %.2f m in %.0f s. Recovering.',
                moved, self.stuck_check_interval)
            if self.nav_state == 1:
                self.deactivate_go_to_point()
            elif self.nav_state == 2:
                self.deactivate_follow_wall()
            self._recover_from_stuck()

    def _recover_from_stuck(self):
        """Back up, rotate ~90° in alternating directions, then restart GoToPoint.
        Blocking is intentional: recovery must complete before navigation resumes."""
        self.wall_dir_index += 1
        self.direction = 'right' if (self.wall_dir_index % 2) else 'left'

        rospy.logwarn('[Bug2] Recovery: backing up...')
        back = Twist()
        back.linear.x = -0.3
        t_end = rospy.get_time() + 1.5
        while rospy.get_time() < t_end and not rospy.is_shutdown():
            self.pub_vel.publish(back)
            rospy.sleep(0.05)

        rospy.logwarn('[Bug2] Recovery: rotating (dir=%s)...', self.direction)
        spin = Twist()
        spin.angular.z = 0.6 if self.direction == 'right' else -0.6
        # π/2 rad at 0.6 rad/s ≈ 2.6 s
        t_end = rospy.get_time() + 2.6
        while rospy.get_time() < t_end and not rospy.is_shutdown():
            self.pub_vel.publish(spin)
            rospy.sleep(0.05)

        self.stop_robot()
        self.obstacle_hit_dist = float('inf')
        # Reset stuck timer so we don't immediately re-trigger.
        self.stuck_check_time  = rospy.get_time()
        self.stuck_check_pos.x = self.position.x
        self.stuck_check_pos.y = self.position.y
        self.start_go_to_point()
        rospy.logwarn('[Bug2] Recovery complete. Resuming GoToPoint.')

    # ------------------------------------------------------------------
    # M-line helpers
    # ------------------------------------------------------------------

    def compute_m_line(self):
        dx = self.goal.x - self.start_position.x
        dy = self.goal.y - self.start_position.y
        if abs(dx) > 1e-6:
            self.m_line_slope     = dy / dx
            self.m_line_intercept = (self.start_position.y
                                     - self.m_line_slope * self.start_position.x)
        else:
            self.m_line_slope     = float('inf')
            self.m_line_intercept = self.start_position.x

    def on_m_line(self):
        if math.isinf(self.m_line_slope):
            return abs(self.position.x - self.m_line_intercept) < self.m_line_threshold
        perp = (abs(self.m_line_slope * self.position.x
                    - self.position.y
                    + self.m_line_intercept) /
                math.sqrt(self.m_line_slope ** 2 + 1))
        return perp < self.m_line_threshold

    def distance_to_goal(self):
        return math.sqrt(
            (self.goal.x - self.position.x) ** 2 +
            (self.goal.y - self.position.y) ** 2)

    # ------------------------------------------------------------------
    # Sub-behaviour activation
    # ------------------------------------------------------------------

    def start_go_to_point(self):
        self.start_position.x = self.position.x
        self.start_position.y = self.position.y
        self.compute_m_line()
        req = MineRescueSetBugStatusRequest()
        req.flag   = True
        req.speed  = self.speed
        req.goal_x = self.goal.x
        req.goal_y = self.goal.y
        try:
            self.client_gtp(req)
            self.change_state(1)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] go_to_point_switch failed: %s', exc)

    def deactivate_go_to_point(self):
        req = MineRescueSetBugStatusRequest()
        req.flag = False
        try:
            self.client_gtp(req)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] go_to_point_switch deactivate: %s', exc)

    def start_follow_wall(self):
        req = MineRescueSetBugStatusRequest()
        req.flag      = True
        req.speed     = self.speed
        req.direction = self.direction
        try:
            self.client_fw(req)
            self.wall_follow_start = rospy.get_time()
            self.change_state(2)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] wall_follower_switch failed: %s', exc)

    def deactivate_follow_wall(self):
        req = MineRescueSetBugStatusRequest()
        req.flag = False
        try:
            self.client_fw(req)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] wall_follower_switch deactivate: %s', exc)

    def stop_robot(self):
        self.pub_vel.publish(Twist())

    def change_state(self, new_state):
        if self.nav_state != new_state:
            rospy.loginfo('[Bug2] State: %s -> %s',
                          self.nav_labels.get(self.nav_state),
                          self.nav_labels.get(new_state))
            self.nav_state = new_state

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def callback_odom(self, msg):
        self.position = msg.pose.pose.position
        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        self.yaw      = transformations.euler_from_quaternion(quat)[2]
        self.got_odom = True

    def callback_laser(self, msg):
        ranges = list(msg.ranges)
        max_r  = msg.range_max
        n      = len(ranges)
        s      = max(1, int(n * 30 / 360))   # ±30° front cone
        clean  = [r if (not math.isnan(r) and
                        not math.isinf(r) and
                        r > 0.15) else max_r
                  for r in ranges]
        # angle_min=-π → index 0=rear, index n//2=forward (angle 0)
        mid = n // 2
        self.laser_front = min(clean[mid - s: mid + s])
        self.got_laser   = True

    def callback_survivor(self, msg):
        rospy.logwarn('[Bug2] Survivor confirmed: ID=%d at (%.1f, %.1f)',
                      msg.survivor_id, msg.position_x, msg.position_y)

    # ------------------------------------------------------------------
    # External homing override service
    # ------------------------------------------------------------------

    def handle_homing_signal(self, req):
        """External homing override (called from a service-callback thread).
        We only mark all survivors found so the main loop's
        check_proximity_detections picks it up safely on its next tick."""
        if req.flag and not self.mission_done:
            rospy.logwarn('[Bug2] External homing signal received.')
            # Mark all survivors so check_proximity_detections redirects cleanly
            for i in range(len(self.survivor_detected)):
                self.survivor_detected[i] = True
            return MineRescueSetBugStatusResponse(
                success=True, message='Homing queued via survivor flags')
        return MineRescueSetBugStatusResponse(
            success=False, message='Flag not set or mission already done')


if __name__ == '__main__':
    try:
        Bug2()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
