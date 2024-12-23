from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from models.main import model_inference
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def download_model():
    # Define the model's local path and the URL to download from
    model_path = "./models/polarity_model/model.safetensors"
    model_url = "https://github.com/himanshuyv/ClarityAI/raw/refs/heads/main/models/polarity_model/model.safetensors?download="

    # Check if the model already exists
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download the model
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            print(f"Failed to download the model. HTTP Status Code: {response.status_code}")
            raise Exception("Model download failed.")
    else:
        print("Model already exists. Skipping download.")

download_model()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    polarity = db.Column(db.String(50), nullable=False)
    concern = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    intensity = db.Column(db.Integer)

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.name
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/chat', methods=['POST'])
def chat():
    
    user_input = request.json.get('message')
    polarity,concern,category,intensity = model_inference(user_input)
    polarity = polarity[0]
    trend_shift_string = ""
    try:
        # fetch the last data from the database
        last_data = Data.query.order_by(Data.id.desc()).first()
        last_polarity = last_data.polarity
        last_category = last_data.category
        lasta_intensity = last_data.intensity


        if last_category != category:
            trend_shift_string += f"Signs of sentiment shift from {last_category} to {category}. "
        if last_polarity != polarity:
            trend_shift_string += f"Polarity has shifted from {last_polarity} to {polarity}. "
            if last_polarity == "Positive" and polarity == "Negative":
                trend_shift_string += "Signs of decline in sentiment has been detected."
            elif last_polarity == "Negative" and polarity == "Positive":
                trend_shift_string += "Signs of improvement in sentiment has been detected."
        elif  last_polarity == polarity:
            if lasta_intensity != intensity:
                if polarity == "Positive":
                    if intensity > lasta_intensity:
                        trend_shift_string += "Intensity of positive sentiment has increased showing improvement."
                    else:
                        trend_shift_string += "Intensity of positive sentiment has decreased showing decline."
                elif polarity == "Negative":
                    if intensity > lasta_intensity:
                        trend_shift_string += "Intensity of negative sentiment has increased showing decline."
                    else:
                        trend_shift_string += "Intensity of negative sentiment has decreased showing improvement."

    except:
        pass
    if concern=='abort':
        trend_shift_string=""
    if trend_shift_string == "":
        trend_shift_string = "No significant trend shift detected."
    if polarity=='Negative':
        if category=='Positive Outlook':
            category='Anxiety'
    bot_response = f"Polarity : {polarity}, Concern : {concern}, Category : {category}, Intensity : {intensity}, Timeline Shift : {trend_shift_string}"
    if concern=='abort':
        bot_response=f"Polarity: {polarity}"
        polarity_value = 1 if polarity == "Positive" else -1 if polarity == "Negative" else 0
        latest_data={'Polarity':polarity_value}
        return jsonify(response=bot_response,latest_data=latest_data)
    new_data = Data(polarity=polarity, concern=concern, category=category, intensity=intensity)
    db.session.add(new_data)
    db.session.commit()

    polarity_value = 1 if polarity == "Positive" else -1 if polarity == "Negative" else 0
    latest_data = {'polarity': polarity_value, 'concern': concern, 'category': category, 'intensity': intensity, 'trend shift': trend_shift_string}
    return jsonify(response=bot_response, latest_data=latest_data)

@app.route('/clear', methods=['POST'])
def clear():
    db.session.query(Data).delete()
    db.session.commit()
    return jsonify(response="Data cleared successfully")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        db.session.query(Data).delete()
        db.session.commit()
    app.run(debug=True)
