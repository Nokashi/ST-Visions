import time
from kafka import KafkaProducer
import json
import pandas as pd
from loguru import logger


def simulate_kafka_stream(
    csv_path,
    topic="test_topic",
    bootstrap_servers="localhost:9092",
    key_field=None,
    delay=0.001
):
    df = pd.read_csv(csv_path).head(20000)
    logger.info(f"Loaded CSV: {len(df)} rows. Streaming to '{topic}'...")

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    records = df.to_dict(orient="records")  #TODO: orient(list)
    for record in records:
        producer.send(topic, value=record)
        logger.info(f"[KAFKA PRODUCER] Sending the following:  {record}")
        producer.flush()
        time.sleep(delay) 

    producer.close()
    logger.info("Test stream finalized")


if __name__ == "__main__":
    simulate_kafka_stream(r'data\unipi_ais_dynamic_2017\unipi_ais_dynamic_dec2017.csv', 'st-viz-topic')