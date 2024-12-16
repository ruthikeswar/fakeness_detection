from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
from sklearn.svm import SVC

class Clustering:
    def __init__(self, res_encodings, int_encodings):
        self.res_encodings = np.array(res_encodings)
        self.int_encodings = np.array(int_encodings)
        # self.total_encodings = np.array(res_encodings + int_encodings), this is for the file optimized_face_reg_dlib.py
        self.total_encodings = np.append(res_encodings, int_encodings, axis=0)

    def dbscan(self):
        dbScan = DBSCAN(eps = 0.4, min_samples=5, metric="euclidean")
        labels = dbScan.fit_predict(self.total_encodings)
        print(f"Number of unique individuals: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print("Cluster labels: ", labels)

        # remove noise -1 labels in the classified labels by dbscan as they are outliers

        valid_indices = [i for i, label in enumerate(labels) if label!=-1]
        filtered_encodings = [self.total_encodings[i] for i in valid_indices]
        filtered_labels = [ labels[i] for i in valid_indices]

        print("Number of clusters: ", len(set(filtered_labels)))
        # Counter function returns the count of unique elements in the given list
        print("Clusters size: ", Counter(filtered_labels))
        return filtered_encodings, filtered_labels

    def svmc(self, fe, fl):

        if len(set(fl)) == 1:
            print("No SVM CLASSIFICATION IS REQUIRED AS ONLY ONE UNIQUE PERSON HAS BEEN DETECTED IN BOTH RESUME AND INTERVIEW")
            return [0 for i in self.res_encodings]
        # FE : filtered_encodings --> contains the encodings of persons present in resume and interview video
        # as resume video encodings are at starting of total_encodings array or filtered encodings array, the LABEL FOR RESUME PERSON WILL BE 0 IN ALL CASES
        # FL : filtered_labels --> contains the labels for persons present in resume video and interview video
        # based on count of unique labels present in filtered labels and frames count ( resume_face_frames, interview_face_frames) we get while extracting the face encodings
        # we calculate the occurance of resume person in the interview video
        X = np.array(fe)
        y = np.array(fl)

        # Train the SVM classifier
        svm_clf = SVC(kernel = 'linear', probability=True)
        svm_clf.fit(X, y)

        # predict and evaluate the classifier
        y_pred = svm_clf.predict(self.res_encodings)
        print("y_pred or resume face label prediction" , y_pred)

        return y_pred
