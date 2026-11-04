docker run --rm -v %CD%:/app hebtts:latest gdown 11NoOJzMLRX9q1C_Q4sX0w2b9miiDjGrv


docker run --gpus all --rm -v %CD%:/app hebtts:latest python infer.py --checkpoint checkpoint.pt --output-dir ./out --text "שלום עולם"