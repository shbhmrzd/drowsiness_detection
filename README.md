# drowsiness_detection

Capturing 120 frames
Time taken : 3.99575185776 seconds
Estimated frames per second : 30.0318949404


For running on local 

-- to get embeddings from the face dataset

python extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7


-- for training the model

python train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle


-- for final video to show all


python inside_camera.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--shape-predictor shape_predictor_68_face_landmarks.dat
