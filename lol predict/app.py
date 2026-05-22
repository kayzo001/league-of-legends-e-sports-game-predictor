import joblib
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

# Initialize the Flask application
app = Flask(__name__)

# =====================================================================
# 1. LOAD MACHINE LEARNING ASSETS
# =====================================================================
print("Initializing Backend: Loading Ultimate Voting Model and Win Rates...")

# Load the trained Pipeline (RobustScaler is embedded inside)
model = joblib.load('models/voting_model.pkl')

# Load the dictionary containing historical champion win rates
champ_win_rates = joblib.load('models/champ_win_rates.pkl')

# Extract a sorted list of champion names to populate the HTML dropdown menus
champions_list = sorted(list(champ_win_rates.keys()))

# =====================================================================
# 2. MAIN APPLICATION ROUTE
# =====================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # ---------------------------------------------------------
            # A. Extract Macro State Features from Form
            # ---------------------------------------------------------
            golddiff = float(request.form['golddiff'])
            platediff = float(request.form['platediff'])
            kda_15 = float(request.form['kda'])
            side_encoded = float(request.form['side_encoded'])
            
            # ---------------------------------------------------------
            # B. Dynamic Feature Engineering: XP Difference
            # ---------------------------------------------------------
            # Aggregate individual player levels
            blue_levels = sum([int(request.form[f'blue_lvl_{role}']) for role in ['top', 'jng', 'mid', 'bot', 'sup']])
            red_levels = sum([int(request.form[f'red_lvl_{role}']) for role in ['top', 'jng', 'mid', 'bot', 'sup']])
            
            # Estimate XP difference (approx. 850 XP per level advantage)
            level_diff = blue_levels - red_levels
            xpdiffat15 = level_diff * 850.0 
            
            # Invert perspective for Red Side
            if side_encoded == 0: 
                xpdiffat15 = -xpdiffat15

            # ---------------------------------------------------------
            # C. Dynamic Feature Engineering: Draft Advantage
            # ---------------------------------------------------------
            # Retrieve selected champions
            blue_champs = [request.form[f'blue_champ_{role}'] for role in ['top', 'jng', 'mid', 'bot', 'sup']]
            red_champs = [request.form[f'red_champ_{role}'] for role in ['top', 'jng', 'mid', 'bot', 'sup']]
            
            # Calculate aggregate team strength
            blue_strength = np.mean([champ_win_rates.get(champ, 0.5) for champ in blue_champs])
            red_strength = np.mean([champ_win_rates.get(champ, 0.5) for champ in red_champs])
            
            # Compute draft differential
            draft_adv = blue_strength - red_strength
            if side_encoded == 0:
                draft_adv = red_strength - blue_strength

            # ---------------------------------------------------------
            # D. Optional Objectives Feature Extraction
            # ---------------------------------------------------------
            # Safely fetch match objectives using .get() to prevent UI crashes
            firstblood = float(request.form.get('firstblood', 0))
            firstdragon = float(request.form.get('firstdragon', 0))
            firstherald = float(request.form.get('firstherald', 0))
            firsttower = float(request.form.get('firsttower', 0))

            # ---------------------------------------------------------
            # E. Model Inference (Prediction)
            # ---------------------------------------------------------
            # Define DataFrame columns EXACTLY as they appear in the Jupyter Notebook
            features_df = pd.DataFrame(
                [[
                    golddiff,       # Maps to 'golddiffat15'
                    xpdiffat15,     # Maps to 'xpdiffat15'
                    platediff,      # Maps to 'platediff'
                    kda_15,         # Maps to 'kda_15'
                    draft_adv,      # Maps to 'draft_adv'
                    side_encoded,   # Maps to 'side_encoded'
                    firstblood,     # Maps to 'firstblood'
                    firstdragon,    # Maps to 'firstdragon'
                    firstherald,    # Maps to 'firstherald'
                    firsttower      # Maps to 'firsttower'
                ]], 
                columns=[
                    'golddiffat15', 'xpdiffat15', 'platediff', 'kda_15', 'draft_adv', 'side_encoded',
                    'firstblood', 'firstdragon', 'firstherald', 'firsttower'
                ]
            )
            
            # Execute pipeline
            pred_class = model.predict(features_df)[0]
            prediction = "Victory" if pred_class == 1 else "Defeat"
            
            # Extract probability
            probability = round(model.predict_proba(features_df)[0][pred_class] * 100, 2)
            
        except Exception as e:
            # Graceful error handling
            prediction = None
            probability = None
            error_message = str(e)

    # Return the template. Passing BOTH 'champs' and 'champions' to guarantee UI compatibility
    return render_template(
        'index.html', 
        champs=champions_list, 
        champions=champions_list, 
        prediction=prediction, 
        probability=probability,
        error=error_message
    )

# =====================================================================
# 3. SERVER EXECUTION
# =====================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)