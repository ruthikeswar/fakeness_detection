import numpy as np
import json


class Final_FakenessScore:
    def __init__(self, video_analysis_json, probability_face_genuine, probability_voice_genuine):
        self.json_file_path = video_analysis_json
        self.probability_face_genuine = probability_face_genuine
        self.probability_voice_genuine = probability_voice_genuine
        with open(self.json_file_path, 'r') as file:
            self.data = json.load(file)
        self.interview_video_duration = 0

    def get_fakeness_score(self, high_likelihood_fakeness_score_threshold = 0.8, medium_likelihood_fakeness_score_threshold = 0.5, low_likelihood_fakeness_score_threshold = 0.5 ):
        self.interview_video_duration = self.data['face_movement_analysis']['total_video_duration']
        if self.probability_face_genuine > 0.85 and self.probability_voice_genuine > 0.87:

            data = self.data
            # Step 1: Total duration (approximation)
            total_duration = self.interview_video_duration  # In seconds

            # Step 2: Compute key metrics
            # forward_bias (head forward duration as a proportion of total duration)
            forward_bias = data['face_movement_analysis']['total_head_forward_duration'] / total_duration

            # iris_engagement_deficit (1 - proportion of iris center duration)
            iris_engagement_deficit = 1 - (
                        data['face_movement_analysis']['total_iris_center_duration'] / total_duration)

            # head_variety_deficit (variance in head movements: left, right, up, down)
            head_durations = [
                data['face_movement_analysis']['total_head_left_duration'],
                data['face_movement_analysis']['total_head_right_duration'],
                data['face_movement_analysis']['total_head_up_duration'],
                data['face_movement_analysis']['total_head_down_duration']
            ]

            head_variety_deficit_normalized = np.var(head_durations) / 10

            # Step 2: Recalculate forward_bias and iris_engagement_deficit to ensure they stay between 0 and 1
            forward_bias_normalized = min(forward_bias, 1)  # Ensuring it does not exceed 1
            iris_engagement_deficit_normalized = min(iris_engagement_deficit, 1)

            W1, W2, W3 = 1, 1, 1  # Weights can be tuned later

            # Step 3: Compute the final fakeness score as a weighted average (still using equal weights here)
            fakeness_score_normalized = (
                                                W1 * forward_bias_normalized +
                                                W2 * iris_engagement_deficit_normalized +
                                                W3 * head_variety_deficit_normalized
                                        ) / (W1 + W2 + W3)

            print(fakeness_score_normalized)
            # Step 4: Re-evaluate classification
            if fakeness_score_normalized > high_likelihood_fakeness_score_threshold:
                fakeness_label_normalized = "High likelihood of faking"
            elif medium_likelihood_fakeness_score_threshold < fakeness_score_normalized <= high_likelihood_fakeness_score_threshold:
                fakeness_label_normalized = "Medium likelihood of faking"
            else:
                fakeness_label_normalized = "Low likelihood of faking"

            return fakeness_label_normalized

        else:
            if self.probability_face_genuine < 0.79 or self.probability_voice_genuine < 0.79:
                return "High likelihood of faking"


