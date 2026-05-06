# InfraWisely

InfraWisely is a MVP for **BESCOM Theme 9: AI for EV Charging Optimization & Infrastructure Planning**.

It is a decision-support dashboard for BESCOM operators and planners. The app predicts EV charging demand by zone and time, detects grid stress, recommends off-peak charging schedules, and ranks priority locations for new EV charging stations.

## Solution Overview

InfraWisely combines a trained EV demand model with grid-aware rule engines:

1. **EV demand forecasting**
   - Uses a saved scikit-learn Random Forest model.
   - Predicts hourly EV charging load in kW for each Bengaluru zone.

2. **Grid stress detection**
   - Adds predicted EV load to base grid load.
   - Compares total load with transformer capacity.
   - Classifies each zone-hour as Low, Medium, High, or Critical risk.

3. **Smart charging optimization**
   - Finds High/Critical evening peak periods.
   - Recommends shifting flexible charging load to off-peak hours.
   - Keeps off-peak load within transformer capacity guardrails.

4. **Station planning**
   - Ranks zones for new EV charging infrastructure.
   - Combines demand, charger gap, traffic, growth, grid health, spare capacity, and stress.
   - Recommends station type and charger count.

5. **Operator dashboard**
   - Next.js + Tailwind + shadcn-style UI.
   - Real map with OpenStreetMap/Leaflet.
   - Dynamic API-driven KPI cards, charts, queue, action plans, and station planning table.




## Local Installation

Prerequisites:

- Python 3.9+
- Node.js 20+
- npm

Clone the repository:

```bash
git clone https://github.com/SandyUndefined/InfraWisely.git
cd InfraWisely
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
npm install
```

## Run Locally

Start the backend in terminal 1:

```bash
cd InfraWisely
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend URLs:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/health
```

Start the frontend in terminal 2:

```bash
cd InfraWisely
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:3000
```

