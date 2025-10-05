"""
st_kafkastream.py - Kafka Stream Implementation

"""

import json
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from st_abstractstream import ST_AbstractStream

class ST_KafkaStream(ST_AbstractStream):
    """
    Kafka-based implementation of ST_AbstractStream.
    Consumes messages from a Kafka topic 

    """

    def __init__(self, topic_name="default-topic", bootstrap_servers="localhost:9092", group_id="stream-group"):
        super().__init__(topic_name)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

        self._assert_topic()

        self.start()

    def _assert_topic(self):
        admin = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        existing = admin.list_topics()

        if self.topic not in existing:
            print(f"Creating topic '{self.topic}'...")
            new_topic = NewTopic(name=self.topic, num_partitions=1, replication_factor=1)
            admin.create_topics(new_topics=[new_topic])
        else:
            print(f"Topic '{self.topic}' exists.")
        admin.close()

    def _connect(self):
        """
        Create KafkaConsumer
        
        """
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            key_deserializer=lambda k: k.decode() if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        )
        print(f"Connected to stream successfully!")
        print(f"---------------------------------")
        print(f"Listening to topic '{self.topic}'...")

    def _poll_data(self):
        """Fetch data from Kafka and forward to visualizer"""
        msg_pack = self.consumer.poll(timeout_ms=500)
        for tp, messages in msg_pack.items():
            for msg in messages:
                value = msg.value
                vessel = value.get("vessel_id", "unknown")
                lon = value.get("lon")
                lat = value.get("lat")
                speed = value.get("speed", "N/A")

                print(f"[KAFKA] Vessel {vessel} @({lon}, {lat}) | Speed: {speed}")
