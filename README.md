# AML_Robust_Deepfake_Detection

# For installing the dependencies and creating the virtual env:
```
1) python3 -m venv ias
2) source ias/bin/activate
3) pip install -r requirements-2.txt
```

# For downloading the actual FaceForensics++ dataset
```
python3 download_faceforensics.py
```

# For preprocessing into frames and storing
```
python3 02_combine_face_extract_RESUME.py

This takes each original video, clips the video such that only face is visible, 
extracts 10 frames per that video, and saves them into a parquet file.

Whenever the systems seems to not handle such processing, press Ctrl+C to halt, 
and it completes the video which was in the middle of preprocessing.

The code can be resumed with the same command.

```

# For performing attacks, comparing results and the rest of work
```
python3 03_malafide_xception_final.py
python3 03A_fgsm_pgd_xception_final.py
python3 04_training_robustness_xception_final.py
python3 05_visualization_analysis_scaled.py
```

Whenever you are free, please run the preprocessing commands to generate the data, since my laptop isn't 
supporting much.




