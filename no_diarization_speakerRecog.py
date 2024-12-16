from no_diarization_split_audio_into_chunks import SplitAudioIntoChunks
from moviepy.editor import VideoFileClip
import speechbrain.pretrained
import torchaudio
from concatenate_audio_files import MergeAudioFiles

# Load the pretrained ECAPA-TDNN model from SpeechBrain
class SpeakerRecognition:

    def __init__(self):
        self.verification_model = speechbrain.pretrained.SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


    def extract_audio_from_video(self, video_path, output_audio_path):
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le', fps=16000)


    def preprocess_audio(self, audio_path):

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if the sample rate is not 16kHz (as required by the model)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        return waveform



    def verify_speaker(self, audio_path1, audio_path2,  threshold=0.8):
        """Verifies if the speakers in the two audio files match."""
        # Load and preprocess audio files
        audio1 = self.preprocess_audio(audio_path1)

        emb1 = self.verification_model.encode_batch(audio1)

        audio_chunks = SplitAudioIntoChunks(audio_path2).create_audio_chunks()


        preprocess_audio_results = []
        for i in audio_chunks:
            waveform = self.preprocess_audio(i)
            preprocess_audio_results.append(waveform)

        embeddings_of_audio_chunks = []

        for j in preprocess_audio_results:
            emb = self.verification_model.encode_batch(j)
            embeddings_of_audio_chunks.append(emb)



        similar_emb_count = 0 # this should be multiplied by 5 secs as each audio chunk is of duration 5 seconds
        similar_emb_duration_chunks = []
        no_speaker = 0 # similarity score will be less than 0.1
        no_speaker_duration_chunks = []
        diff_speaker = 0 # similarity core will be between the 0.1 and 0.5
        diff_speaker_duration_chunks = []
        diff_speaker_audio_file_indexes = []
        k = 0
        for i in range(len(embeddings_of_audio_chunks)):
            similarity_score = self.verification_model.similarity(emb1, embeddings_of_audio_chunks[i])
            print(f"{k} - {k+5} --> {similarity_score.mean()}")


            if similarity_score.mean() >= 0.6:
                similar_emb_count +=1
                similar_emb_duration_chunks.append(f"{k} - {k+5}")
            elif (similarity_score.mean() < 0.5) and (similarity_score.mean() > 0.15):
                diff_speaker += 1
                diff_speaker_duration_chunks.append(f"{k} - {k+5}")
                diff_speaker_audio_file_indexes.append(i)
            elif (similarity_score.mean() < 0.15) or ((similarity_score.mean() > 0.5) and (similarity_score.mean() < 0.6)):

                no_speaker += 1
                no_speaker_duration_chunks.append(f"{k} - {k+5}")
            k += 5


        diff_speaker_audio_chunks = []
        for a in diff_speaker_audio_file_indexes:
            diff_speaker_audio_chunks.append(audio_chunks[a])




        if diff_speaker == 0:
            voice_flag = True
            voice_result_msg = "The speakers in both videos are the same."
            diff_speaker_voice_file = None
            speakerDiarizationAnalyis = {
                "Resume_Speaker_Duration_In_Interview": f"{(similar_emb_count * 5)+5} seconds" ,
                "Resume_Person_Duration_Chunks":similar_emb_duration_chunks,
                "Diff_Person_Duration_Chunks": diff_speaker_duration_chunks,
                "No_Speaker_Duration_Chunks": no_speaker_duration_chunks,
                "probability_of_genuine": 1
            }
            return voice_flag, voice_result_msg, speakerDiarizationAnalyis, diff_speaker_voice_file

        else:
            voice_flag = False

            if similar_emb_count == 0:
                voice_result_msg = "The speakers in both videos are different."
                diff_speaker_voice_file = audio_path2
                speakerDiarizationAnalyis = {
                    "Resume_Speaker_Duration_In_Interview": 0,
                    "Resume_Person_Duration_Chunks": similar_emb_duration_chunks,
                    "Diff_Person_Duration_Chunks": diff_speaker_duration_chunks,
                    "No_Speaker_Duration_Chunks": no_speaker_duration_chunks,
                    "probability_of_genuine" : 0
                }
            else:
                voice_result_msg = "There are multiple speakers in the video."
                diff_speaker_voice_file = MergeAudioFiles(diff_speaker_audio_chunks).concatenate_audio_files()
                speakerDiarizationAnalyis = {
                    "Resume_Speaker_Duration_In_Interview": f"{(similar_emb_count * 5)+5} seconds" ,
                    "Resume_Person_Duration_Chunks": similar_emb_duration_chunks,
                    "Diff_Person_Duration_Chunks": diff_speaker_duration_chunks,
                    "No_Speaker_Duration_Chunks": no_speaker_duration_chunks,
                    "probability_of_genuine":((similar_emb_count*5)+5)/(((similar_emb_count*5)+5) + ((diff_speaker*5)-5)),
                }

            return voice_flag, voice_result_msg, speakerDiarizationAnalyis, diff_speaker_voice_file



    # Main function to run the verification
    def main(self,video_path1, video_path2):
        # Paths for temporary audio files
        audio_path1 = video_path1.split('.')[0]+".wav"
        audio_path2 = video_path2.split('.')[0]+".wav"

        # Extract audio from both videos
        self.extract_audio_from_video(video_path1.split(".")[0]+".mp4", audio_path1)
        self.extract_audio_from_video(video_path2.split(".")[0]+".mp4", audio_path2)
        x,y,z, k = self.verify_speaker(audio_path1, audio_path2)
        return x,y,z,k





# DEBUGGING PURPOSE
# SpeakerRecognition().main("Self_Intro.mp4", "Genuine_with_voice.mp4")

