import os
import subprocess


class WebmToMp4:


    def __init__(self, video_file):
        self.video_file = video_file


    def webm_to_mp4(self):
        input_file = self.video_file
        windows_input_file = subprocess.run(
            ["wslpath", "-w", input_file], capture_output=True, text=True
        ).stdout.strip()

        # in order to write the file into mp4 there are some problems with path
        # if we want our video to be written at local directory which is in path format of wsl, we need to convert it to windows format
        # plain wsl path will not be accessed or recognized.


        output_file = windows_input_file.split('.webm')[0]+".mp4"

        print("after converting it into mp4 it will get saved at : ", output_file)
        ffmpeg_path = os.path.join('/mnt','c', 'ffmpeg', 'ffmpeg.exe') # this is the path when i am running from wsl windows subsystem for linux
        print(f"webm to .mp4 conversion path {ffmpeg_path}")
        # FFmpeg command to convert .webm to .mp4
        result = subprocess.run([
            ffmpeg_path,
            '-i', windows_input_file,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            output_file
        ])
