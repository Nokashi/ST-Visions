'''
    st_vizstream.py - v2025.09.27

    Authors: Paraschos Moraitis, Andreas Tritsarolis
'''

import threading
import time
import json
from abc import ABC, abstractmethod
from queue import Queue

from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic

class ST_AbstractStream(ABC):

    def __init__(self, topic_name: str):
        self.topic = topic_name
        self.consumer = None
        self._thread = None
        self._stop = False
        self.data_queue = Queue()
        self.max_queue_size = None


    @abstractmethod
    def _connect(self):
        """
        Abstract method that initializes a given stream connection from a provider (e.g Kafka)
        """
        pass

    @abstractmethod
    def _poll_data(self):
        """
        Abstract method for data polling and processing
        """
        pass


    def _consume_loop(self):
        """
        Data consumption loop for a given stream
        """
        try:
            self._connect()
        except Exception as e:
            print(f"[STREAM ERROR] Failed to connect: {e}")
            return 

        try:
            while not self._stop:
                self._poll_data()
                time.sleep(0.05)
        except Exception as e:
            print(f"[STREAM ERROR] {e}")
        finally:
            self._close()


    def start(self):
        """
        Start background thread
        
        """
        if self._thread and self._thread.is_alive():
            print("Stream already running.")
            return

        self._stop = False
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        print("Stream thread initialized")

    def stop(self):
        """
        Stop background thread
        
        """
        print("Stopping stream...")
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("Stream stopped")

    def _close(self):
        """
        Thread cleanup method
        
        """
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass



class ST_KafkaStream(ST_AbstractStream):
    """
    Kafka-based implementation of ST_AbstractStream.
    Consumes messages from a Kafka topic 

    Default batch size is 1000

    """

    def __init__(self, topic_name="default-topic", bootstrap_servers="localhost:9092", group_id="stream-group", max_queue_size = 10000):
        super().__init__(topic_name)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.max_queue_size = max_queue_size

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
        Initialize the Kafka Consumer
        
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

    def get_stream_data(self, max_points=500):
        """
        Drain the queue and return all records as a columnar dict (orient='list').
        Each record is consumed exactly once.
        Only returns up to the last `max_points` records.
        """
        if self.data_queue.empty():
            return None

        records = []
        while not self.data_queue.empty():
            records.append(self.data_queue.get())

        if not records:
            return None

        data = {k: [] for k in records[0].keys()}
        for record in records:
            for k, v in record.items():
                data[k].append(v)

        for k in data.keys():
            data[k] = data[k][-max_points:]

        return data

    def _poll_data(self):
        msg_pack = self.consumer.poll(timeout_ms=500)
        for tp, messages in msg_pack.items():
            for msg in messages:
                value = msg.value
                lon = value.get("lon")
                lat = value.get("lat")

                if lon is None or lat is None:
                    continue

                # insert to queue check 
                if self.data_queue.qsize() >= self.max_queue_size:
                    _ = self.data_queue.get()
                self.data_queue.put(value)
            