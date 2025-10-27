# run_server.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from login_page import login_app
from main_app import main_app

app = FastAPI(title="Music Classifier with Auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# mount /login và /app
app = gr.mount_gradio_app(app, login_app, path="/login")
app = gr.mount_gradio_app(app, main_app,  path="/app")

if __name__ == "__main__":
    uvicorn.run("run_server:app", host="0.0.0.0", port=8000, reload=True)
