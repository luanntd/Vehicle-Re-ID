# Vehicle-Re-ID
## Tasks
- Finetune YOLO for detection task
- Run Pipeline with Kafka, Spark successfully
- Adjust Kafka, Spark code for display result like `run_reid.py`

## Install packages
Make venv with the same python version set in PYSPARK_PYTHON path.
```bash
pip install -r requirements.txt
```

## Run without Kafka, Spark
Edit camera path (1, 2) and result save dir in `run_reid.py`
```bash
python run_reid.py
```

## Run with Kafka, Spark
Run Kafka server, cd to kafka dir and run
```bash
bin\windows\kafka-server-start.bat config\server.properties
```

Run the consumer
```bash
python Consumer.py --topic cam1 --topic-2 cam2 --save-dir results
```

Run the producer for camera 1
```bash
python Producer.py --topic cam1 --camera path/to/camere1
```

Run the producer for camera 2
```bash
python Producer.py --topic cam2 --camera path/to/camere2
```
