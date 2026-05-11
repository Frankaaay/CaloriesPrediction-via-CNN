# Industrialization Guide: Calories Prediction Pipeline

This document explains the 4 phases required to transition a standalone Data Science/Machine Learning script into a production-grade, "Industrial-Level" software service. This is exactly what engineers at tech companies do to deploy models.

---

## Phase 1: Build a REST API (FastAPI)
**What it does:** 
Instead of loading the heavy PyTorch model every time a user wants to predict an image, we create a continuously running backend server. It exposes a web endpoint (URL) that applications can send data to.

**How it works:**
FastAPI creates a `POST /predict` endpoint. When an image is received via an HTTP request, the backend processes the image through the CNN, generates the calorie prediction, and returns the result as a lightweight JSON object (e.g., `{"calories": 450}`).

**How it's implemented in our project:**
We will create `src/api.py`. When you run this script, it loads `model.pth` into memory exactly once. It uses `uvicorn` (a lightning-fast web server) to listen for incoming images on port 8000. 

---

## Phase 2: Refactor Frontend (Streamlit)
**What it does:** 
It separates the "User Interface" from the "Brain". This is known as "Decoupling" or "Separation of Concerns." 

**How it works:**
Currently, our `app.py` has to import PyTorch, load the model, and run the inference itself. If 1,000 people use the app, it crashes. In a decoupled architecture, the frontend only handles showing the website to the user. When a user uploads a picture, the frontend simply acts as a messenger: it forwards the image to the FastAPI backend and displays the JSON response to the user.

**How it's implemented in our project:**
We will strip all PyTorch and Machine Learning code out of `app.py`. Instead, `app.py` will use the Python `requests` library to send a POST request to `http://localhost:8000/predict`.

---

## Phase 3: Containerization (Docker)
**What it does:** 
It packages the entire application (code, Python version, dependencies, OS libraries) into a standardized unit called a "Container".

**How it works:**
Have you ever heard "It works on my machine, I don't know why it's broken on yours"? Docker solves this. By writing a set of instructions (`Dockerfile`), Docker builds a mini-virtualized environment that contains everything needed to run your app. 

**How it's implemented in our project:**
We will create:
1. `backend.Dockerfile` - Bundles PyTorch, FastAPI, and `model.pth`
2. `frontend.Dockerfile` - Bundles Streamlit and `app.py`
3. `docker-compose.yml` - A master file that links the backend and frontend together. You'll be able to type `docker-compose up` and immediately start the entire project on any computer in the world without installing Python.

---

## Phase 4: CI/CD Pipeline (GitHub Actions)
**What it does:** 
Continuous Integration / Continuous Deployment (CI/CD) automates the testing and deployment of your code. 

**How it works:**
Every time you push a new change to GitHub, a server in the cloud automatically downloads your code and runs your tests. If the tests fail, it blocks the code from being merged or deployed.

**How it's implemented in our project:**
We will create a `.github/workflows/ci.yml` file. It will tell GitHub: "Every time Frank pushes code, install the requirements and run `pytest tests/test_data_loader.py`." When recruiters look at your GitHub repository, they will see a green "Passing" badge, proving you write reliable, tested software.