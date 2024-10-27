from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import sys
sys.path.append('..')
from models.main import model_inference

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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

    bot_response = f"Polarity = {polarity}, Concern = {concern}, Category = {category}, Intensity = {intensity}"
    # new_data = Data(polarity=polarity, concern=concern, category=category, intensity=intensity)
    # db.session.add(new_data)
    # db.session.commit()

    polarity_value = 1 if polarity == "Positive" else -1 if polarity == "Negative" else 0
    latest_data = {'polarity': polarity_value, 'concern': concern, 'category': category, 'intensity': intensity}
    return jsonify(response=bot_response, latest_data=latest_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
