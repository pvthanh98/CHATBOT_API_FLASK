from flask import Flask, render_template, session
from flask import jsonify
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
import vietnameseChatbot as chatbot
app = Flask(__name__)
#-*- coding: utf-8 -*-
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins="*")

if __name__ == '__main__':
    socketio.run(app)


@socketio.on("connect")
def connect():
    if(session.get("messages")==None):
        emit("connected","ok")
    else:
        emit("connected",session.get("messages"))
        
@socketio.on("message")
def messageReceive(data):
    data =data.encode('latin1').decode('utf8')
    msg = chatbot.message_response(data)
    print(data)
    emit("message", msg)

@app.route("/")
def index():
	return render_template("chat.html")

