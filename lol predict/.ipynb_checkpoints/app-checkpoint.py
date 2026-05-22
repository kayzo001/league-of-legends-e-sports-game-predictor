from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML Assets
try:
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    champ_win_rates = joblib.load('models/champ_win_rates.pkl')
    champions_list = sorted(list(champ_win_rates.keys()))
except Exception as e:
    print(f"Startup Warning: Models not found ({e}). Run the Jupyter Notebook first!")
    champions_list = ["Aatrox", "Ahri", "Akali"]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    
    if request.method == 'POST':
        try:
            # 1. Macro State
            golddiff = float(request.form['golddiff'])
            platediff = float(request.form['platediff'])
            kda = float(request.form['kda'])
            side_encoded = float(request.form['side_encoded'])
            
            # 2. Calculate XP Difference from Levels
            blue_levels = sum([int(request.form[f'blue_lvl_{role}']) for role in ['top', 'jng', 'mid', 'bot', 'sup']])
            red_levels = sum([int(request.form[f'red_lvl_{role}']) for role in ['top', 'jng', 'mid', 'bot', 'sup']])
            
            level_diff = blue_levels - red_levels
            xpdiff = level_diff * 850.0 
            if side_encoded == 0: # Flip perspective for Red Side
                xpdiff = -xpdiff

            # 3. Calculate Draft Advantage from Champions
            blue_champs = [request.form[f'blue_champ_{role}'] for role in ['top', 'jng', 'mid', 'bot', 'sup']]
            red_champs = [request.form[f'red_champ_{role}'] for role in ['top', 'jng', 'mid', 'bot', 'sup']]
            
            blue_strength = np.mean([champ_win_rates.get(champ, 0.5) for champ in blue_champs])
            red_strength = np.mean([champ_win_rates.get(champ, 0.5) for champ in red_champs])
            
            draft_adv = blue_strength - red_strength
            if side_encoded == 0:
                draft_adv = red_strength - blue_strength

            # 4. Predict
            input_data = np.array([[golddiff, xpdiff, platediff, kda, draft_adv, side_encoded]])
            scaled_data = scaler.transform(input_data)
            
            pred_class = model.predict(scaled_data)[0]
            prediction = "Victory" if pred_class == 1 else "Defeat"
            probability = round(model.predict_proba(scaled_data)[0][pred_class] * 100, 2)
            
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, probability=probability, champs=champions_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)