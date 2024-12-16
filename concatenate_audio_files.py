from pydub import AudioSegment

class MergeAudioFiles:
    def __init__(self, list_of_audio_paths):
        self.audio_paths = list_of_audio_paths
        print("Audio Chunk paths list", self.audio_paths)

    def concatenate_audio_files(self):
        concatenated_audio_path = self.audio_paths[1].split('/')[:-1]
        diff_speaker_voice_file = "/".join(concatenated_audio_path)+"/diffSpeakerAudioFile.wav"

        combined_audio = AudioSegment.empty()

        for file in self.audio_paths:
            audio = AudioSegment.from_file(file)
            combined_audio+=audio

        combined_audio.export(diff_speaker_voice_file, format="wav")
        return diff_speaker_voice_file