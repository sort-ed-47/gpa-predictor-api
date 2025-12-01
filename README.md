📘 SortED GPA Prediction API
AI-powered GPA prediction model for the SortED platform, built using FastAPI, RandomForest, SHAP explainability, and a synthetic student dataset.
This API predicts a student's GPA using 5 performance indicators and also provides explainability for every prediction.
🚀 Features
✅ GPA Prediction API
Predicts GPA using:
Attendance Score
Assignment Score
Event Score
Academic Score
SPI (SortED Performance Index)
✅ Explainability API (SHAP)
Get feature-wise SHAP values showing:
Why the predicted GPA increased/decreased
Which factors had the strongest impact
✅ FastAPI Auto Docs
Interactive API docs at:
/docs
✅ Ready for Deployment
Includes:
Dockerfile
requirements.txt
main.py
Model files (.pkl)
📊 Input Features & Ranges
Feature	Description	Range
attendance_score	Attendance ratio	0.40 → 1.00
assignment_score	Assignment submission rate	0.00 → 1.00
event_score	Event participation	0.00 → 1.00
academic_score	Academic marks / 100	40 → 94
spi	SortED Performance Index	0.22 → 0.99
SPI Formula:
spi = (0.35 * attendance_score
     + 0.25 * assignment_score
     + 0.20 * event_score
     + 0.20 * (academic_score / 100))
🧠 Model Details
Algorithm: RandomForestRegressor
Dataset size: 500 synthetic students
Preprocessing: StandardScaler
Evaluation:
RMSE ≈ 0.50–0.70
R² ≈ 0.75–0.90 depending on seed
Model Files:
gpa_predictor_model.pkl
gpa_predictor_scaler.pkl
📁 Project Structure
api/
 ├── main.py
 ├── requirements.txt
 ├── Dockerfile
 ├── gpa_predictor_model.pkl
 └── gpa_predictor_scaler.pkl
⚙️ Installation (Local Run)
1. Clone the Repo
git clone https://github.com/<your-username>/sorted-gpa-api.git
cd sorted-gpa-api
2. Install Requirements
pip install -r requirements.txt
3. Start the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
4. Open API Docs
Visit:
http://localhost:8000/docs
📡 API Endpoints
🔷 1. Predict GPA
POST /predict
Request Body
{
  "attendance_score": 0.92,
  "assignment_score": 0.86,
  "event_score": 0.40,
  "academic_score": 78,
  "spi": 0.73
}
Response
{
  "predicted_gpa": 7.82
}
🔷 2. Explain Prediction (SHAP)
POST /explain
Response Example
{
  "predicted_gpa": 7.82,
  "shap_explanation": [
    {
      "feature": "attendance_score",
      "value": 0.92,
      "shap_value": 0.18
    },
    {
      "feature": "assignment_score",
      "value": 0.86,
      "shap_value": 0.05
    }
  ]
}
🐳 Docker Deployment
1. Build Image
docker build -t sorted-gpa-api .
2. Run Container
docker run -p 8000:8000 sorted-gpa-api
☁️ Render Deployment (Free)
1. Push project to GitHub
2. Go to https://render.com
3. New → Web Service
4. Connect GitHub repo
5. Choose Dockerfile as runtime
6. Deploy 🚀
Your public API will look like:
https://sorted-gpa-api.onrender.com/docs
🤝 Contributing
Feel free to submit issues or pull requests.
📬 Contact
MatriDev Technologies
Developer: Shivansh Prasad
Email: youremail@example.com
Product: SortED
