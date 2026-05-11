from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import stats, charts, probability, prediction

app = FastAPI(title="Student Placement Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats.router,       prefix="/api")
app.include_router(charts.router,      prefix="/api")
app.include_router(probability.router, prefix="/api")
app.include_router(prediction.router,  prefix="/api")

@app.get("/")
def root():
    return {"message": "Student Placement Analytics API is running"}
