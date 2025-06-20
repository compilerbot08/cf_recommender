🚀 Codeforces Problem Recommendation System

A machine learning-based personalized recommendation system for Codeforces users to improve their practice by suggesting unsolved problems tailored to their skill level and weaknesses.

🎯 Features
🔎 Personalized Recommendations — Suggests 10 unsolved problems at a time based on your rating, problem tags, and past submission history.

🏷️ Tag-wise Practice — Prioritizes problems from tags you frequently get wrong, helping strengthen weak topics.

📊 Machine Learning Model — Uses Random Forest Classifier with >90% prediction accuracy.

⚖️ Explore/Exploit Strategy — Balances easier problems with challenging ones near your rating.

🌐 Interactive Frontend — Built with React.js and React Router DOM for smooth navigation.

📂 Live Integration with Codeforces API — Real-time fetching of problems and user submissions.

🖥️ Tech Stack
Frontend	Backend	Machine Learning	Deployment (Optional)
React.js	Flask	scikit-learn	Vercel (Frontend) / Render (Backend)

⚙️ How It Works
Enter your Codeforces handle.

The system fetches your submission history via Codeforces API.

Features are engineered using:

Problem rating

User rating

Problem tags

Count of wrong submissions

The RandomForestClassifier predicts your probability of success on unattempted problems.

10 problems recommended → based on predicted success and tag-wise weaknesses.

📌 Example Screenshot (Optional)
(Add a screenshot or demo gif if available)

🚀 Setup Instructions
1️⃣ Backend Setup
bash
Copy
Edit
git clone https://github.com/your-username/codeforces-recommender.git
cd backend
pip install -r requirements.txt
python app.py
2️⃣ Frontend Setup
bash
Copy
Edit
cd frontend
npm install
npm start
🌟 Upcoming Features
🔗 Recommend Similar Problems for any individual problem (using KMeans clustering)

📈 Track progress across tags/topics

🖼️ User authentication (future scope)

✨ Demo
Live URL (if deployed): [https://cf-recommender-frontend.vercel.app/](https://cf-recommender-frontend.vercel.app/)
