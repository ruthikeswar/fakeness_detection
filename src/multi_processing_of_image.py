import cv2
import face_recognition
from multiprocessing import Pool, cpu_count, Manager
from datetime import datetime

class MultiProcessing_Images:
    def __init__(self, video_path):
        self.video_path = video_path
        self.fps = None
        self.two_person_frames = None  # This will be initialized as a managed list

    def process_frame(self, frame_data):
        """Extract face encodings from a single frame."""
        frame_number, frame, two_person_frames_shared = frame_data
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        img_path_list = self.video_path.split('/')[:-1]
        img_path = "/".join(img_path_list)

        if len(face_locations) > 1 and len(two_person_frames_shared) < 11:
            total_image_path = img_path + f'/{int(frame_number // self.fps)}_{int(frame_number % self.fps)}.jpg'
            cv2.imwrite(total_image_path, frame)
            two_person_frames_shared.append(total_image_path)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return frame_number, face_encodings

    def video_to_frames(self):
        """Extract frames from the video."""
        vid_to_frame_start = datetime.now()
        print("video to frame conversion has been started at:", datetime.now())
        cap = cv2.VideoCapture(self.video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second
        frames = []
        frame_number = 0
        print("if video is getting opened", cap.isOpened())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process 1 frame every second (to reduce workload)
            if frame_number % self.fps == 0:
                frames.append((frame_number, frame))
            frame_number += 1

        cap.release()
        vid_to_frame_end = datetime.now()
        print('Time to taken to convert the video to frames is:', (vid_to_frame_end - vid_to_frame_start), "secs")
        print('\n')
        return frames

    def main(self, num_processes):
        frames = self.video_to_frames()
        encoding_start = datetime.now()
        print("Extraction of encodings of face from the frames has been started at:", datetime.now())

        count_of_frames = len(frames)
        strt = 0
        end = 499
        encodings = []
        if count_of_frames == 0:
            return encodings

        # Initialize Manager for shared data
        with Manager() as manager:
            self.two_person_frames = manager.list()  # Shared list managed by Manager

            while count_of_frames > 500:
                start = datetime.now()
                with Pool(num_processes) as pool:
                    results = pool.map(
                        self.process_frame,
                        [(frame_number, frame, self.two_person_frames) for frame_number, frame in frames[strt:end]]
                    )
                encodings.append(results)
                count_of_frames -= 500
                print("500 frames faces encodings extraction is done.", (datetime.now() - start), "secs")
                strt = end
                end += 500
            else:
                start = datetime.now()
                with Pool(num_processes) as pool:
                    results = pool.map(
                        self.process_frame,
                        [(frame_number, frame, self.two_person_frames) for frame_number, frame in frames[strt:]]
                    )
                encodings.append(results)
                print(f"{count_of_frames} frames faces encodings extraction is done.", (datetime.now() - start), "secs")

            # Combine results
            encoding_end = datetime.now()
            print(f"Time taken to extract the encodings of all frames {len(encodings[0])}:",
                  (encoding_end - encoding_start), "secs")
            total_encodings = [i[1] for i in encodings[0]]
            print(f"The two person frames are saved at location: {list(self.two_person_frames)}")
            return total_encodings, list(self.two_person_frames)

if __name__ == '__main__':
    video_resume_path = "<Path to video>"  # Replace with your video path

    num_processes = cpu_count()  # Use all available CPU cores
    processor = MultiProcessing_Images(video_resume_path)

    # Get encodings
    encodings, two_person_frames = processor.main(num_processes)
    print("Encodings:", encodings)
    print("Two person frames:", two_person_frames)
