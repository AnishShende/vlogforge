"""Tests for warning propagation in the orchestrator."""
import pytest
from datetime import datetime
import asyncio
from app.models import JobStatus, WSProgressEvent
from app.tasks.orchestrator import jobs_db, websockets_db, broadcast_progress

@pytest.mark.anyio
async def test_warning_routing_to_job_status_and_ws():
    """Ensure warnings in JobStatus are properly propagated via WebSocket."""
    job_id = "test-warning-job"
    
    # Setup mock job
    jobs_db[job_id] = JobStatus(
        job_id=job_id,
        status="running",
        progress=50,
        message="Running...",
        project_id="p1",
        created_at=datetime.utcnow(),
        warnings=["Test warning 1", "Test warning 2"]
    )
    
    # Mock WebSocket
    class MockWebSocket:
        def __init__(self):
            self.sent_events = []
            
        async def send_text(self, data: str):
            self.sent_events.append(data)
            
    mock_ws = MockWebSocket()
    websockets_db[job_id] = {mock_ws}
    
    try:
        # Trigger broadcast
        await broadcast_progress(job_id, "processing", 60, "Testing warnings")
        
        # Verify event was sent
        assert len(mock_ws.sent_events) == 1
        
        # Parse the JSON string
        import json
        payload = json.loads(mock_ws.sent_events[0])
        
        # Assert WSProgressEvent schema matches expectations
        assert payload["stage"] == "processing"
        assert payload["progress"] == 60
        assert "warnings" in payload
        assert len(payload["warnings"]) == 2
        assert payload["warnings"][0] == "Test warning 1"
        assert payload["warnings"][1] == "Test warning 2"
        
    finally:
        # Cleanup
        if job_id in jobs_db:
            del jobs_db[job_id]
        if job_id in websockets_db:
            del websockets_db[job_id]
