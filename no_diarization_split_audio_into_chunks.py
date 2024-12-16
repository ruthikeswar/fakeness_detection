from pydub import AudioSegment

class SplitAudioIntoChunks:


    def __init__(self, interview_audio_file_path): #change the path according to the paths coming from api's or wsl or from any side

        self.interviewAudioPath = interview_audio_file_path

    def get_audio_duration(self, file_path):
        audio = AudioSegment.from_file(file_path)
        duration_seconds = int(len(audio) / 1000)  # Duration in seconds
        return duration_seconds

    def split_audio(self,file_path, start_time, end_time, output_file):
        """
        Splits an audio file from start_time to end_time and saves the result.

        Args:
            file_path (str): Path to the input audio file.
            start_time (int): Start time in milliseconds.
            end_time (int): End time in milliseconds.
            output_file (str): Path to save the split audio file.
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_file(file_path)

            # Extract the segment
            split_segment = audio[start_time:end_time]

            # Export the split audio
            split_segment.export(output_file, format="wav")
        except Exception as e:
            print(f"Error splitting audio: {e}")




    def create_audio_chunks(self):
        '''
        start_ms = 10000  # Start at 10 seconds (in milliseconds)
        end_ms = 20000  # End at 20 seconds (in milliseconds)
        output_path = "output_audio.wav"
        '''

        # takes a full length audio file as an input and gives output of small audio files of duration 10 secs


        audio_file = self.interviewAudioPath
        interview_audio_name = self.interviewAudioPath.split(".")[0]
        audio_duration = self.get_audio_duration(audio_file)

        output_files = []
        for i in range(0, audio_duration, 5):
            if i + 5 < audio_duration:
                self.split_audio(audio_file, i*1000, (i+5)*1000, f'{interview_audio_name}_{i}.wav') # we need to give start and end in milliseconds
                output_files.append(f'{interview_audio_name}_{i}.wav')
            else:
                self.split_audio(audio_file, (i * 1000), audio_duration * 1000, f'{interview_audio_name}_{i}.wav')
                output_files.append(f'{interview_audio_name}_{i}.wav')

        return output_files