from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    response = db.Column(db.String(500), nullable=False)

# Routes
@app.route('/')
def root():
    return render_template('index.html')

@app.route('/index')
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
            return "Login Failed"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('root'))

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
    bot_response = f"Bot: You said '{user_input}'"

    # Save to Chat History
    if 'user_id' in session:
        user_id = session['user_id']
        chat_history = ChatHistory(user_id=user_id, message=user_input, response=bot_response)
        db.session.add(chat_history)
        db.session.commit()

    return jsonify(response=bot_response)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return jsonify({'history': []})
    
    user_id = session['user_id']
    history = ChatHistory.query.filter_by(user_id=user_id).all()
    history_data = [{'message': chat.message, 'response': chat.response} for chat in history]

    return jsonify(history=history_data)

@app.route('/load_chat/<int:chat_id>')
def load_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({'messages': []})
    
    chat_history = ChatHistory.query.filter_by(id=chat_id, user_id=session['user_id']).all()
    messages = [{'text': chat.message, 'sender': 'user'} for chat in chat_history]
    messages += [{'text': chat.response, 'sender': 'bot'} for chat in chat_history]
    return jsonify({'messages': messages})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
