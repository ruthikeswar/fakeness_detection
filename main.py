import boto3
import os
from fastapi import FastAPI, File, UploadFile, Form


from Modular_faceMotion import FaceMotionDetection
from optimized_with_multi_processing import Dlib_FaceRecognition

from no_diarization_speakerRecog import SpeakerRecognition
from final_decision import Final_FakenessScore

app = FastAPI()

# AWS S3 Client
s3 = boto3.client('s3')



# Helper functions
def parse_s3_path(s3_path):
    """
    Parse the S3 path into bucket name and key.
    Example: s3://bucket-name/folder/file -> bucket-name, folder/file
    """
    s3_components = s3_path.replace("s3://", "").split("/", 1)
    print(f"bucketname and folder path : {s3_components[0]}, {s3_components[1]}")
    return s3_components[0], s3_components[1]


def download_from_s3(bucket_name, s3_key):
    """
    Download a video file from S3 and save it locally.
    """
    video_local_path = os.path.join("/tmp", bucket_name+"_"+s3_key.split('-')[0].replace('/', "_")+".webm")
    print(f"the downloaded video got saved at : {video_local_path}")
    s3.download_file(bucket_name, s3_key, video_local_path)
    return video_local_path


def process_with_face_motion(video_path):

    # Your faceMotion.py logic comes here (adjust as needed for the correct paths and variables)
    video_path = video_path.split('.')[0]+".mp4" # changing the extension from webm to mp4
    print("The facemotion detection module object is getting initialized with the following video_path", video_path)
    fmd = FaceMotionDetection(video_path)
    print("The process_video method is called")
    output_json_path, output_video_path = fmd.process_video()

    return output_json_path, output_video_path


def upload_to_s3(file_path, bucket_name, s3_key):
    """
    Upload a file to S3 and return the file URL.
    """
    s3.upload_file(file_path, bucket_name, s3_key)
    return f"s3://{bucket_name}/{s3_key}"

@app.post("/process-video/")
async def process_video( resume_s3_bucket_path: str = Form(None), interview_s3_bucket_path: str = Form(None)):
    interview_video_path = None
    resume_video_path = None

    # If the video is provided via S3 bucket
    if interview_s3_bucket_path:
        bucket_name, s3_key = parse_s3_path(interview_s3_bucket_path)
        interview_video_path = download_from_s3(bucket_name, s3_key)

    if resume_s3_bucket_path:
        bucket_name, s3_key = parse_s3_path(resume_s3_bucket_path)
        resume_video_path = download_from_s3(bucket_name, s3_key)

    # If the video is uploaded directly


    if not resume_video_path:
        return {"error": "No valid video provided"}

    # Check if video file exists
    if not interview_video_path:
        return {"error": "No valid video provided"}

    # face module execution:

    print("resume_video_path:", resume_video_path)
    print("interview_video_path: ", interview_video_path)
    face_flag, face_result, total_analysis, two_person_frame_paths = Dlib_FaceRecognition(resume_video_path, interview_video_path).main()

    # voice module execution:
    voice_flag, voice_result, voice_diarization, diff_speaker_voice_file = SpeakerRecognition().main(resume_video_path, interview_video_path)


    # Track eye gaze and head movements module:
    output_json_path, output_video_path = process_with_face_motion(interview_video_path)

    # final decision model :
    probability_face_genuine = total_analysis[0]["probability_of_genuine"]
    probability_voice_genuine = voice_diarization["probability_of_genuine"]

    final_fakenessScore_obj = Final_FakenessScore(output_json_path, probability_face_genuine, probability_voice_genuine)
    final_decision = final_fakenessScore_obj.get_fakeness_score()


    '''commenting the below code as of now we dont have the permissions to upload the files into the aws s3
    # Upload the result files to S3
    if interview_s3_bucket_path:
        bucket_name, s3_key = parse_s3_path(interview_s3_bucket_path)
        s3_output_json_path = upload_to_s3(output_json_path, bucket_name, os.path.join("/tmp", bucket_name+"_"+s3_key.split('-')[0].replace('/', "_")+"_analysis.json"))
        s3_output_video_path = upload_to_s3(output_video_path, bucket_name, os.path.join("/tmp", bucket_name+"_"+s3_key.split('-')[0].replace('/', "_")+"_analysis.mp4"))
        return {
            "output_json_s3_path": s3_output_json_path,
            "output_video_s3_path": s3_output_video_path,
        }
        
    '''

    # Return local paths if no S3 path was given
    return {
        "face_module_outputs":{'face_flag': face_flag, 'face_result': face_result, 'total_analysis': total_analysis, "two_person_images": two_person_frame_paths },
        "voice_module_outputs": {'voice_flag': voice_flag, 'voice_result': voice_result, 'voice_analysis': voice_diarization, 'diff_speaker_audio_file': diff_speaker_voice_file},
        "Head&Iris_Module_Outputs":{
        "interviewVid_output_json_path": output_json_path,
        "interviewVid_output_video_path": output_video_path},
        "final_decision":final_decision
    }
