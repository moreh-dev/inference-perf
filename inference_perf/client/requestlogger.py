# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from datetime import datetime
from typing import Optional
import logging
import threading

logger = logging.getLogger(__name__)


class RequestLogger:
    """Logs requests to a file in JSON Lines format (one JSON object per line)."""

    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize the RequestLogger.

        Args:
            save_path: Path to save requests. If None, generates a default path.
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = f"requests-{timestamp}.jsonl"
        
        self.save_path = save_path
        self.file_handle = None
        self.lock = threading.Lock()
        self.request_count = 0
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        if dir_path:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Directory created/verified: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
                raise
        
        # Open file in append mode
        try:
            self.file_handle = open(save_path, "a", encoding="utf-8")
            logger.info(f"Request logger initialized. Saving requests to: {save_path}")
            # Write a header comment to verify file is writable
            self.file_handle.write(f"# Request log started at {datetime.now().isoformat()}\n")
            self.file_handle.flush()
        except Exception as e:
            logger.error(f"Failed to open request log file {save_path}: {e}", exc_info=True)
            raise

    def log_request(
        self,
        request_data: str,
        stage_id: int,
        scheduled_time: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Log a request to the file.

        Args:
            request_data: JSON string of the request payload
            stage_id: Stage ID for this request
            scheduled_time: When the request was scheduled
            timestamp: Current timestamp (optional, defaults to now)
        """
        if self.file_handle is None:
            return

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        try:
            # Parse request_data to get structured info
            request_payload = json.loads(request_data)
            
            log_entry = {
                "request_id": self.request_count,
                "timestamp": timestamp,
                "scheduled_time": scheduled_time,
                "stage_id": stage_id,
                "payload": request_payload,
            }
            
            with self.lock:
                json.dump(log_entry, self.file_handle, ensure_ascii=False)
                self.file_handle.write("\n")
                self.file_handle.flush()
                self.request_count += 1
                if self.request_count % 100 == 0:
                    logger.debug(f"Logged {self.request_count} requests so far")
                
        except Exception as e:
            logger.error(f"Failed to log request: {e}", exc_info=True)

    def close(self) -> None:
        """Close the file handle."""
        if self.file_handle:
            with self.lock:
                self.file_handle.close()
                self.file_handle = None
            logger.info(f"Request logger closed. Total requests logged: {self.request_count}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

