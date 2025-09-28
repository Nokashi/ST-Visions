'''
    st_vizstream.py - v2025.09.27

    Authors: Paraschos Moraitis, Andreas Tritsarolis
'''


import json
import threading
import time
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import geopandas as gpd
from shapely import wkt

from st_visions.st_visualizer import st_visualizer
import st_visions.express as viz_express

class st_vizstream:
    def __init__(self, topic_name="default-topic", bootstrap_servers="localhost:9092", group_id="stream-group"):

        self.topic = topic_name
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

        self.plot = None

        self.consumer = None
        self._thread = None
        self._stop = False


        self._assert_topic()
        self.start()

    def _assert_topic(self):
        admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        existing_topics = admin_client.list_topics()
        if self.topic not in existing_topics:
            print(f"Creating topic '{self.topic}'...")
            topic = NewTopic(name=self.topic, num_partitions=1, replication_factor=1)
            admin_client.create_topics(new_topics=[topic])
        else:
            print(f"Topic '{self.topic}' exists.")
        admin_client.close()

    def _consume_data(self):
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            key_deserializer=lambda k: k.decode() if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )

        print(f"Listening to {self.topic}'...")

        try:
            while not self._stop:
                msg_pack = self.consumer.poll(timeout_ms=500)
                for tp, messages in msg_pack.items():
                    for msg in messages:
                        print("Message received:")
                        print(f"  Key: {msg.key}")
                        print(f"  Value: {msg.value}")
                        print("-" * 40)
                time.sleep(0.05)
        except Exception as e:
            print(f"Consumer error: {e}")
        finally:
            self.consumer.close()
            print(" Kafka consumer stopped.")

    def start(self):
        if self._thread and self._thread.is_alive():
            print("Consumer already running.")
            return
        self._stop = False
        self._thread = threading.Thread(target=self._consume_data, daemon=True)
        self._thread.start()
        print("Consumer thread started")

    def stop(self):
        print("Stopping consumer...")
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("Consumer stopped")
