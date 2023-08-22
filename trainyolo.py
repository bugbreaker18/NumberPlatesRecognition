import subprocess
import time

def trainedyolo():
    start_time = time.time()

    command = [
        'python',
        'train.py',
        '--workers', '2',
        '--img', '640',
        '--batch', '16',
        '--epochs', '100',
        '--data', 'data/plates.yaml',
        '--weights', 'yolov5s.pt',
        '--device', 'cuda:0',  # Use 'cpu' or 'cuda:0' as needed
        '--cache'
    ]

    subprocess.run(command, cwd='yolov5')

    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {(end_time-start_time):.2f}')