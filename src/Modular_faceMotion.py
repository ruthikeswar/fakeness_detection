# IMPORTING THE LIBRARIES


import mediapipe as mp
import cv2
import math
import os
import numpy as np
import json


# importing the class self.face_motion_analysis to calculate the video_timestamps

from faceMotionCalculation import FaceMotionAnalysis

class FaceMotionDetection:
    def __init__(self, video_file):
        # the below three variables can be changed according to paths we get
        self.video_file = video_file # total path
        self.video_file_name = video_file.split("/")[-1].split(".")[0] # video file name

        self.face_motion_analysis = FaceMotionAnalysis()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.FPS = None
        self.width = None
        self.height = None
        self.fourcc = None
        self.out = None
        self.frame_num = None
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.R_E_RIGHTMOST = [33]
        self.R_E_LEFTMOST = [133]
        self.L_E_RIGHTMOST = [362]
        self.L_E_LEFTMOST = [263]
        self.nose_2d = None
        self.nose_3d = None



    def save_to_json(self):
        result = {
            'face_motion_records': self.face_motion_analysis.face_motion_records,
            'face_movement_analysis': {
                'total_head_left_duration': self.face_motion_analysis.total_head_left_duration,
                'total_head_right_duration': self.face_motion_analysis.total_head_right_duration,
                'total_head_up_duration': self.face_motion_analysis.total_head_up_duration,
                'total_head_down_duration': self.face_motion_analysis.total_head_down_duration,
                'total_head_forward_duration': self.face_motion_analysis.total_head_forward_duration,
                'total_iris_center_duration': self.face_motion_analysis.total_iris_center_duration,
                'total_iris_right_duration': self.face_motion_analysis.total_iris_right_duration,
                'total_iris_left_duration': self.face_motion_analysis.total_iris_left_duration,
                'total_video_duration': self.face_motion_analysis.total_video_duration
            }
        }
        print(self.video_file)
        json_file_path = os.path.join('/tmp', self.video_file_name+"_analysis.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

    # Calculate Euclidean distance between two points
    def euclidean_distance(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    # Determine the Iris Position
    def iris_position(self, iris_center, right_point, left_point):
        center_to_right_distance = self.euclidean_distance(iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)
        ratio = center_to_right_distance / total_distance
        if ratio <= 0.42:
            return "left"
        elif 0.42 <= ratio <= 0.57:
            return "center"
        else:
            return "right"

    # Process each frame of the video
    def process_video(self):

        outputVideo_file_path = os.path.join('/tmp', self.video_file_name + "_analysis.mp4")

        print(f"the video file to be opened by cv is : {self.video_file}")
        self.cap = cv2.VideoCapture(self.video_file)
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(outputVideo_file_path,self.fourcc, self.FPS, (self.width, self.height))
        print("If video can be read,opened or not", self.cap.isOpened())
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            results = self.face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                # Draw circles around the iris
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)

                # Calculate iris position
                iri_pos = self.iris_position(center_right, mesh_points[self.R_E_RIGHTMOST][0], mesh_points[self.R_E_LEFTMOST][0])
                self.face_motion_analysis.update_iris_left_movement(iri_pos, self.frame_num, self.FPS)
                self.face_motion_analysis.update_iris_right_movement(iri_pos, self.frame_num, self.FPS)
                self.face_motion_analysis.update_iris_center_movement(iri_pos, self.frame_num, self.FPS)

                cv2.putText(image, f"IRIS POSITION : {iri_pos}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # to find the head position in 3d
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            if idx == 1:
                                self.nose_2d = (lm.x * img_w, lm.y * img_h)
                                self.nose_3d = (lm.x * img_w, lm.y* img_h, lm.z* 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    # Calculate head position
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * self.width
                    cam_matrix = np.array(
                        [[focal_length, 0, self.height / 2], [0, focal_length, self.width / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
                    if y < -10:
                        text, d = "Left ", "l"
                    elif y > 8:
                        text, d = "Right ", "r"
                    elif x < -10:
                        text, d = "Head is Down", "d"
                    elif x > 10:
                        text, d = "Head is Up", "u"
                    else:
                        text, d = "Forward", "f"

                    nose_3d_projection, jacobian = cv2.projectPoints(self.nose_3d, rot_vec, trans_vec, cam_matrix,dist_matrix)

                    p1 = (int(self.nose_2d[0]), int(self.nose_2d[1]))
                    p2 = (int(self.nose_2d[0] + y * 10), int(self.nose_2d[1] - x * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                    # Add text on the image

                    cv2.putText(image, f"HEAD POSITION : {text}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)
                    self.face_motion_analysis.update_head_left_movement(d, self.frame_num, self.FPS)
                    self.face_motion_analysis.update_head_right_movement(d, self.frame_num, self.FPS)
                    self.face_motion_analysis.update_head_up_movement(d, self.frame_num, self.FPS)
                    self.face_motion_analysis.update_head_down_movement(d, self.frame_num, self.FPS)
                    self.face_motion_analysis.update_head_forward_movement(d, self.frame_num, self.FPS)

            self.out.write(image)
            # cv2.imshow('Head Pose Estimation', image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to stop # here it is 16 am gonna change to 1 to check runtime
                self.face_motion_analysis.update_head_left_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_head_right_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_head_down_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_head_up_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_head_forward_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_iris_left_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_iris_right_movement("x", self.frame_num, self.FPS)
                self.face_motion_analysis.update_iris_center_movement("x", self.frame_num, self.FPS)
                break
        self.face_motion_analysis.update_head_left_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_head_right_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_head_down_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_head_up_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_head_forward_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_iris_left_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_iris_right_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.update_iris_center_movement("x", self.frame_num, self.FPS)
        self.face_motion_analysis.total_video_duration_calc(self.frame_num, self.FPS)

        self.save_to_json()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        print("video processing is done this is printed to get the video duration and if the flow is coming here or not")
        video_duration = int(self.frame_num / self.FPS)
        print(f"The interview video duration: ", video_duration)

        output_video_path = os.path.join("/tmp", self.video_file_name+"_analysis.mp4") # these output paths can be change according to the need
        output_json_path = os.path.join("/tmp", self.video_file_name+"_analysis.json")

        return output_json_path, output_video_path

# the actual path we get after we upload the video in the endpoint/api - 'C:\Users\RUTHIK~1.T\AppData\Local\Temp\talentwiz.webm')


























