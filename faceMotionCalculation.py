
# DEFINING THE CLASS TO STORE THE ANY FACE MOTION DURATIONS


class FaceMotionAnalysis:
    def __init__(self):
        self.face_motion_records = {
            'head_left_info': [],
            'head_right_info': [],
            'head_up_info':[],
            'head_down_info':[],
            'head_forward_info':[],
            'iris_right_info':[],
            'iris_left_info':[],
            'iris_center_info':[]
        }

        # head variables

        # head movements
        self.head_left_start_time = None
        self.head_right_start_time = None
        self.head_up_start_time = None
        self.head_down_start_time = None
        self.head_forward_start_time = None

        # aggregated head movements
        self.total_head_left_duration = 0
        self.total_head_right_duration = 0
        self.total_head_up_duration = 0
        self.total_head_down_duration = 0
        self.total_head_forward_duration = 0

        # eye movements
        self.iris_left_start_time = None
        self.iris_right_start_time = None
        self.iris_center_start_time = None

        # aggregated eye movements
        self.total_iris_left_duration = 0
        self.total_iris_right_duration = 0
        self.total_iris_center_duration = 0

        # total duration of video
        self.total_video_duration = 0

    # func to convert seconds(int) into minutes(time)
    def convert_int_to_minutes(self, num):
        minutes = num // 60
        seconds = num % 60

        return f"{int(minutes)}:{int(seconds)}"



    # head-left

    def update_head_left_movement(self, text, frame_num, fps):

        if text == "l":
            if self.head_left_start_time is None:
                self.head_left_start_time = frame_num/fps
        else:
            if self.head_left_start_time is not None:
                movement_duration = (frame_num/fps) - self.head_left_start_time

                if int(movement_duration)>=1:
                    self.total_head_left_duration+= movement_duration
                    start_time = self.head_left_start_time
                    end_time = frame_num/fps
                    self.face_motion_records['head_left_info'].append({
                        "start_time":self.convert_int_to_minutes(start_time),
                        "end_time":self.convert_int_to_minutes(end_time),
                        "duration":int(movement_duration)
                    })
                self.head_left_start_time = None

    # head-right

    def update_head_right_movement(self,text,frame_num, fps):

        if text == "r":
            if self.head_right_start_time is None:
                self.head_right_start_time = frame_num/fps
        else:
            if self.head_right_start_time is not None:
                movement_duration = (frame_num/fps)- self.head_right_start_time

                if int(movement_duration) >= 1:
                    self.total_head_right_duration += movement_duration
                    start_time =self.head_right_start_time
                    end_time = frame_num/fps
                    self.face_motion_records['head_right_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": int(movement_duration)
                    })
                self.head_right_start_time = None

    # head-up

    def update_head_up_movement(self,text, frame_num, fps):

        if text == "u":
            if self.head_up_start_time is None:
                self.head_up_start_time = frame_num/fps
        else:
            if self.head_up_start_time is not None:
                movement_duration = (frame_num/fps) -self.head_up_start_time

                if int(movement_duration) >= 1:
                    self.total_head_up_duration += movement_duration
                    start_time = self.head_up_start_time
                    end_time = frame_num/fps
                    self.face_motion_records['head_up_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": int(movement_duration)
                    })
                self.head_up_start_time = None

    # head-down
    def update_head_down_movement(self, text, frame_num, fps):

        if text == "d":
            if self.head_down_start_time is None:
                self.head_down_start_time = frame_num/fps
        else:
            if self.head_down_start_time is not None:
                movement_duration = (frame_num/fps) - self.head_down_start_time

                if int(movement_duration) >= 1:
                    self.total_head_down_duration += movement_duration
                    start_time = self.head_down_start_time
                    end_time = frame_num/fps
                    self.face_motion_records['head_down_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": int(movement_duration)
                    })
                self.head_down_start_time = None

            # head-forward
    def update_head_forward_movement(self, text, frame_num, fps):

        if text == "f":
            if self.head_forward_start_time is None:
                self.head_forward_start_time = frame_num/fps
        else:
            if self.head_forward_start_time is not None:
                movement_duration = (frame_num/fps) - self.head_forward_start_time

                if int(movement_duration) >= 1:
                    self.total_head_forward_duration += movement_duration
                    start_time = self.head_forward_start_time
                    end_time = frame_num/fps
                    self.face_motion_records['head_forward_info'].append({
                                "start_time": self.convert_int_to_minutes(start_time),
                                "end_time": self.convert_int_to_minutes(end_time),
                                "duration": int(movement_duration)
                            })
                self.head_forward_start_time = None

    # iris_left

    def update_iris_left_movement(self, text, frame_num, video_fps):

        if text == "left":
            if self.iris_left_start_time is None:
                self.iris_left_start_time = frame_num / video_fps

        else:
            if self.iris_left_start_time is not None:
                movement_duration = (frame_num / video_fps) - self.iris_left_start_time

                if int(movement_duration) >= 2:
                    self.total_iris_left_duration += movement_duration
                    start_time = self.iris_left_start_time
                    end_time = frame_num / video_fps
                    self.face_motion_records['iris_left_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": movement_duration
                    })
                self.iris_left_start_time = None

    # iris_right

    def update_iris_right_movement(self, text, frame_num, video_fps):

        if text == "right":
            if self.iris_right_start_time is None:
                self.iris_right_start_time = frame_num / video_fps

        else:
            if self.iris_right_start_time is not None:
                movement_duration = (frame_num/video_fps) - self.iris_right_start_time

                if int(movement_duration) >= 2:
                    self.total_iris_right_duration += movement_duration
                    start_time = self.iris_right_start_time
                    end_time = frame_num/video_fps
                    self.face_motion_records['iris_right_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": movement_duration
                    })
                self.iris_right_start_time = None


    # iris_center

    def update_iris_center_movement(self, text, frame_num, video_fps):

        if text == "center":
            if self.iris_center_start_time is None:
                self.iris_center_start_time = frame_num / video_fps

        else:
            if self.iris_center_start_time is not None:
                movement_duration = (frame_num / video_fps) - self.iris_center_start_time

                if int(movement_duration) >= 2:
                    self.total_iris_center_duration += movement_duration
                    start_time = self.iris_center_start_time
                    end_time = frame_num / video_fps
                    self.face_motion_records['iris_center_info'].append({
                        "start_time": self.convert_int_to_minutes(start_time),
                        "end_time": self.convert_int_to_minutes(end_time),
                        "duration": movement_duration
                    })
                self.iris_center_start_time = None


    def total_video_duration_calc(self,frame_num, video_fps):
        self.total_video_duration = int(frame_num/video_fps)