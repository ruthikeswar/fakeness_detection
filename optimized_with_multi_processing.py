from multiprocessing import Pool, cpu_count
from multi_processing_of_image import MultiProcessing_Images
from Dbscan_SVMC import Clustering
from collections import Counter
from datetime import datetime
import numpy as np
from webm_to_mp4 import WebmToMp4

class Dlib_FaceRecognition:
    def __init__(self, video_resume_path, video_interview_path):

        # we need to change the video extensions from webm to mp4
        WebmToMp4(video_resume_path).webm_to_mp4()
        WebmToMp4(video_interview_path).webm_to_mp4()

        self.video_resume_path = video_resume_path.split(".")[0]+".mp4"
        self.video_interview_path = video_interview_path.split(".")[0]+".mp4"



    def process(self, encodings):

        valid_encodings = []
        two_person_count = 0
        zero_person_count = 0
        for i in encodings:
            if len(i) == 1:
                valid_encodings.append(i)
            elif len(i) == 2:
                two_person_count += 1
            else:
                zero_person_count += 1
        valid_encodings = np.array([encoding[0] for encoding in valid_encodings])
        return two_person_count, valid_encodings


    def main(self):


        # Extract faces from both videos
        num_cores = cpu_count()

        res_encodings, x = MultiProcessing_Images(self.video_resume_path).main(num_cores)
        two_person_frame_countR, resume_faces = self.process(res_encodings)

        interview_encodings, two_person_frame_paths = MultiProcessing_Images(self.video_interview_path).main(num_cores)
        two_person_frame_countI, interview_faces = self.process(interview_encodings)

        # frames count
        print(f"Number of frames extracted from resume video: {len(resume_faces) + two_person_frame_countR}") # if we want, we can consider adding noPersonDetectedFrames also.
        print(f"Number of frames extracted from interview video: {len(interview_faces) + two_person_frame_countI}") # if we want, we can consider adding noPersonDetectedFrames also.

        # single face detected frames of resume and interview
        print(f" Single person face frames detected in the resume video: {len(resume_faces)}")
        print(f"Single person face frames detected in the interview video: {len(interview_faces)}")


        # clustering
        print('\n')
        clus_start = datetime.now()
        print("Clustering and svmClassification started at: ", datetime.now())
        clus_classification = Clustering(resume_faces, interview_faces)
        clus_enc, clus_labels = clus_classification.dbscan()
        res_predicted_labels = clus_classification.svmc(clus_enc, clus_labels)
        clus_end = datetime.now()
        print("Time taken to cluster the whole resume and interview face encodings and predicting the classifcation labels by svm: ", (clus_end - clus_start), "secs")
        print('\n')


        # resume_single_person_count, interview_single_person_count, clus_labels, res_predicted_labels

        print('\n')
        dec_making_start = datetime.now()
        print("Decision making by using clusteringlabels and resumepredicted labels by svm: ", datetime.now())

        total_clus_labels_dict = Counter(clus_labels)
        res_predicted_labels = Counter(res_predicted_labels)
        for i in res_predicted_labels:
            RES_LABEL = i
            total_clus_labels_dict[i] -= res_predicted_labels[i]

        interview_clus_labels = total_clus_labels_dict

        print(f"two person frame count in the resume : {two_person_frame_countR}")
        print(f"two person frame count in the interview: {two_person_frame_countI}")

        if two_person_frame_countI ==0 :

            if len(interview_clus_labels) == 1:

                # if no person has been detected in the interview video:

                if len(interview_faces)!=0:
                    face_flag = True
                    face_result_msg = "The same person appears in both videos."
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "resume_person_durationPercentage_in_interview (person0)":(total_clus_labels_dict[RES_LABEL]/ len(interview_faces)) * 100,
                        "probability_of_genuine":(total_clus_labels_dict[RES_LABEL]/ len(interview_faces))
                    }]
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths
                else:
                    face_flag = False
                    face_result_msg ="No face has been detected in the interview video"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "resume_person_durationPercentage_in_interview (person0)": 0,
                        "probability_of_genuine": 0
                    }]
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths

            elif len(interview_clus_labels) == 2:
                face_flag = False
                if total_clus_labels_dict[RES_LABEL] == 0:
                    face_result_msg = f" The person in resume video and interview video doesn't match, they are totally different"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "resume_person_durationPercentage_in_interview (person0)": 0,
                        "probability_of_genuine": 0
                    }]
                    for label, count in interview_clus_labels.items():
                        if label == RES_LABEL:
                            pass
                        else:
                            total_analysis[0][f'person{label}_durationPercentage_in_interview(diff. person)'] = (total_clus_labels_dict[label] / len(interview_faces)) * 100
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths
                else:
                    face_result_msg = f" '{len(interview_clus_labels)}' persons has been detected in the interview video"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                    }]
                    for label, count in interview_clus_labels.items():
                        if label == RES_LABEL:
                            total_analysis[0][f'resume_person_durationPercentage_in_interview (person{label})'] = \
                                (total_clus_labels_dict[label] / len(interview_faces)) * 100
                        else:
                            total_analysis[0][f'person{label}_durationPercentage_in_interview(diff_per_from_interview)'] = (total_clus_labels_dict[label] / len(interview_faces)) * 100
                    total_analysis[0]["probability_of_genuine"] = total_analysis[0]['resume_person_durationPercentage_in_interview (person0)']/100
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths



            else:
                face_flag = False

                face_result_msg = f" '{len(interview_clus_labels)}' persons has been detected in the interview video"
                total_analysis = [{
                    "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                }]
                for label, count in interview_clus_labels.items():
                    if label == RES_LABEL:
                        total_analysis[0][f'resume_person_durationPercentage_in_interview (person{label})'] = \
                            (total_clus_labels_dict[label] / len(interview_faces)) * 100
                    else:
                        total_analysis[0][f'person{label}_durationPercentage_in_interview (diff_per_from_interview)'] = (total_clus_labels_dict[label] / len(
                            interview_faces)) * 100

                total_analysis[0]["probability_of_genuine"] = total_analysis[0][
                    'resume_person_durationPercentage_in_interview (person0)']/100
                dec_making_end = datetime.now()
                print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                return face_flag, face_result_msg, total_analysis, two_person_frame_paths

        else:
            if len(interview_clus_labels) == 1:

                # if no person has been detected in the interview video:

                if len(interview_faces) != 0:
                    face_flag = False
                    face_result_msg = "The same person who have been in video resume has been detected in interview video and  ALSO ANOTHER PERSON OTHER THAN VIDEO RESUME PERSON HAVE BEEN DETECTED."
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "resume_person_durationPercentage_in_interview (person0)":(total_clus_labels_dict[RES_LABEL] / (len(interview_faces) + two_person_frame_countI)) * 100,
                        "two_person_frame_count": two_person_frame_countI,
                        "two_person_frame_occ_perc(%)": (two_person_frame_countI/(len(interview_faces) + two_person_frame_countI)) * 100,
                    }]
                    total_analysis[0]["probability_of_genuine"] = total_analysis[0]['resume_person_durationPercentage_in_interview (person0)']/100
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths
                else:
                    face_flag = False
                    face_result_msg = "No single person frame has been detected in the interview video, in all frames two persons have been present"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "resume_person_durationPercentage_in_interview (person0})": 0,
                        "Two_person_frame_count": two_person_frame_countI,
                        "Two_person_frame_occ_per(%)": (two_person_frame_countI/ (len(interview_faces)+two_person_frame_countI)) * 100
                    }]
                    total_analysis[0]["probability_of_genuine"] = total_analysis[0][
                        'resume_person_durationPercentage_in_interview (person0)']
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths

            elif len(interview_clus_labels) == 2:
                face_flag = False

                if total_clus_labels_dict[RES_LABEL] == 0:
                    face_result_msg = f" Different person and two_person_frames have been detected in the interview video"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "two_person_frames_count": two_person_frame_countI,
                        "two_person_frame_occ_per(%)": (two_person_frame_countI / (len(interview_faces) + two_person_frame_countI)) * 100,
                        "resume_person_durationPercentage_in_interview (person0)":0
                    }]
                    total_analysis[0]["probability_of_genuine"] = total_analysis[0]['resume_person_durationPercentage_in_interview (person0)']
                    for label, count in interview_clus_labels.items():
                        if label == RES_LABEL:
                            pass
                        else:
                            total_analysis[0][
                                f'person{label}_durationPercentage_in_interview (diff_per_from_interview)'] = \
                                (total_clus_labels_dict[label] / (len(
                                interview_faces) + two_person_frame_countI)) * 100
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths

                else:
                    face_result_msg = f" Different_person and Resume-person and two_person_frames have been detected in the interview video"
                    total_analysis = [{
                        "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                        "two_person_frames_count": two_person_frame_countI,
                        "two_person_frame_occ_per(%)": (two_person_frame_countI / (
                                    len(interview_faces) + two_person_frame_countI)) * 100
                    }]
                    for label, count in interview_clus_labels.items():
                        if label == RES_LABEL:
                            total_analysis[0][f'resume_person_durationPercentage_in_interview (person{label})'] = \
                                (total_clus_labels_dict[label] / (len(interview_faces) + two_person_frame_countI)) * 100
                        else:
                            total_analysis[0][
                                f'person{label}_durationPercentage_in_interview (diff_per_from_interview)'] = \
                                (total_clus_labels_dict[label] / (len(
                                interview_faces) + two_person_frame_countI)) * 100
                    total_analysis[0]["probability_of_genuine"] = total_analysis[0]['resume_person_durationPercentage_in_interview (person0)']/100
                    dec_making_end = datetime.now()
                    print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                    return face_flag, face_result_msg, total_analysis, two_person_frame_paths



            else:
                face_flag = False

                face_result_msg = f" '{len(interview_clus_labels)}' persons  and two_person_frames have been detected in the interview video"
                total_analysis = [{
                    "total_interview_frames": len(interview_faces) + two_person_frame_countI,
                    "two_person_frames_count": two_person_frame_countI,
                    "Two_person_frame_occ_per(%)": (two_person_frame_countI+ (len(interview_faces) + two_person_frame_countI)) * 100
                }]
                for label, count in interview_clus_labels.items():
                    if label == RES_LABEL:
                        total_analysis[0][f'resume_person_durationPercentage_in_interview (person{label})'] = \
                            (total_clus_labels_dict[label] / (len(interview_faces) + two_person_frame_countI))*100
                    else:
                        total_analysis[0][f'person{label}_durationPercentage_in_interview (diff_per_from_interview)'] = (total_clus_labels_dict[label] / (len(
                            interview_faces) + two_person_frame_countI))*100

                total_analysis[0]["probability_of_genuine"] = total_analysis[0][
                    'resume_person_durationPercentage_in_interview (person0)']/100
                dec_making_end = datetime.now()
                print("Time taken to make decision: ", (dec_making_end - dec_making_start), "secs")
                return face_flag, face_result_msg, total_analysis, two_person_frame_paths

