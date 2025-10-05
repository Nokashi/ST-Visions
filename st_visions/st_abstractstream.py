"""
st_abstractstream.py v2025.05.09

Authors: Paraschos Moraitis, Andreas Tritsarolis


Abstract streaming class for handling any kind of stream data source. Handles the lifecycle 

"""

import threading
import time
from abc import ABC, abstractmethod
from queue import Queue

class ST_AbstractStream(ABC):

    def __init__(self, topic_name: str):
        self.topic = topic_name
        self.consumer = None
        self._thread = None
        self._stop = False
        self.data_queue = Queue()


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
