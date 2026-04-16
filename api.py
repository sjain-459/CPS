from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import asyncio
from main import run_simulation_stream
import json
import os

app = FastAPI(title="CTMAS Simulation API")

# Serve the 'results' directory to show plots on the frontend
if not os.path.exists("results"):
    os.makedirs("results")
app.mount("/results", StaticFiles(directory="results"), name="results")

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Run simulation as an async iterator. Since run_simulation_stream is synchronous, 
        # normally we should run it in a threadpool to not block the asyncio loop,
        # but since it's a generator we can just loop over it and insert async sleeps.
        # This is a safe non-blocking mechanism for simple demonstration.
        
        for event_data in run_simulation_stream():
            await websocket.send_text(json.dumps(event_data))
            
            # Artificial sleep to make the frontend simulation visually pleasing and steady
            event_type = event_data.get("event")
            if event_type == "client_training" and event_data.get("status") == "training":
                await asyncio.sleep(0.5) # Time for node training
            elif event_type == "sensor_stream":
                await asyncio.sleep(0.05) # Fast streaming sensor data
            else:
                await asyncio.sleep(0.2)
                
    except Exception as e:
        print(f"WebSocket client disconnected or error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
