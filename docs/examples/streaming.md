# Streaming Examples

## Basic JSON Streaming

```python
from smartsurge import Client, JSONStreamingRequest

client = Client(base_url="https://api.example.com")

# Stream a large JSON response

result, history = client.stream_request(
    streaming_class=JSONStreamingRequest,
    endpoint="/large-dataset",
    chunk_size=4096  # 4KB chunks
)

# Process the JSON data

print(f"Received {len(result)} items")
print(f"First item: {result}")
```

## Resumable Streaming with State File

```python
import os
from smartsurge import Client, JSONStreamingRequest

client = Client(base_url="https://api.example.com")

state_file = "download_state.json"

# Check if we're resuming a previous download

is_resuming = os.path.exists(state_file)
print(f"Resuming download: {is_resuming}")

try:
# Stream a large JSON response with resumability
    result, history = client.stream_request(
        streaming_class=JSONStreamingRequest,
        endpoint="/very-large-dataset",
        params={"limit": 50000},
        chunk_size=8192,
        state_file=state_file
    )

    print(f"Download completed successfully!")
    print(f"Received {len(result)} items")
    
    # Clean up state file after successful download
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"Removed state file")
    except Exception as e:
        print(f"Download interrupted: {e}")
        print("You can resume the download later by running the script again")
```

## Custom Streaming Request Implementation

```python
from smartsurge import Client, AbstractStreamingRequest
import csv
import io

class CSVStreamingRequest(AbstractStreamingRequest):
"""A streaming request implementation that processes CSV data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_rows = []
        
    def start(self):
        """Start the streaming request."""
        # Add Range header if resuming
        if self.position > 0:
            self.headers['Range'] = f'bytes={self.position}-'
            
        # Make the request with appropriate retry settings
        self.logger.debug(f"Starting CSV streaming request to {self.endpoint}")
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[^2_500][^2_502][^2_503][^2_504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            with session.get(
                self.endpoint,
                headers=self.headers,
                params=self.params,
                stream=True,
                timeout=(10, 30)
            ) as self.response:
                if not self.response.ok:
                    raise StreamingError(f"Request failed with status {self.response.status_code}")
                
                # Get content length if available
                if 'Content-Length' in self.response.headers:
                    try:
                        self.total_size = int(self.response.headers['Content-Length'])
                    except (ValueError, TypeError):
                        pass
                
                # Process chunks
                for chunk in self.response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        self.process_chunk(chunk)
                
                self.completed = True
                
        except requests.RequestException as e:
            self.save_state()
            raise StreamingError(f"CSV streaming request failed: {e}")
    
    def resume(self):
        """Resume the streaming request from saved state."""
        state = self.load_state()
        if not state:
            raise ResumeError("Failed to load state for resumption")
        
        self.start()
    
    def process_chunk(self, chunk):
        """Process a chunk of CSV data."""
        self.accumulated_data.extend(chunk)
        self.position += len(chunk)
        
        # Log progress for large responses
        if self.total_size and self.total_size > self.chunk_size * 10:
            progress = (self.position / self.total_size) * 100
            if self.position % (self.chunk_size * 10) < len(chunk):
                self.logger.debug(f"Download progress: {progress:.1f}%")
        
        # Save state periodically
        if self.state_file and self.position % (self.chunk_size * 10) < len(chunk):
            self.save_state()
    
    def get_result(self):
        """Parse the accumulated data as CSV and return rows."""
        if not self.completed:
            raise StreamingError("CSV streaming request not completed")
        
        try:
            # Parse CSV data
            csv_text = self.accumulated_data.decode('utf-8')
            csv_file = io.StringIO(csv_text)
            reader = csv.DictReader(csv_file)
            return list(reader)
        except Exception as e:
            raise StreamingError(f"Failed to parse CSV data: {e}")
    
# Use the custom streaming class

client = Client(base_url="https://api.example.com")

# Stream a large CSV file

csv_data, history = client.stream_request(
    streaming_class=CSVStreamingRequest,
    endpoint="/exports/large-report.csv",
    state_file="csv_download_state.json"
)

# Process the CSV data

print(f"Downloaded {len(csv_data)} rows of CSV data")
for row in csv_data[:5]:  # Print first 5 rows
    print(row)
```
