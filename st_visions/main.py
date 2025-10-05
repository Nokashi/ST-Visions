"""
main.py - Example entrypoint
"""

import time
from st_kafkastream import ST_KafkaStream

if __name__ == "__main__":
    stream = ST_KafkaStream(
        topic_name="ais-topic",
        group_id="ais-group"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
