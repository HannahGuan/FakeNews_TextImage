@echo off
REM This script trains BLIP Model v2 with full training data
REM and saves all output to a log file while showing real-time progress

python3 main.py --classification-type 6_way --mode train --batch-size 8 --lr 0.0001 --dropout 0.3 --pooling mean --epochs 10 --eval-after-train

python3 main.py --classification-type 3_way --mode train --batch-size 8 --lr 0.0001 --dropout 0.3 --pooling mean --epochs 10 --eval-after-train

python3 main.py --classification-type 2_way --mode train --batch-size 8 --lr 0.0001 --dropout 0.3 --pooling mean --epochs 10 --eval-after-train
