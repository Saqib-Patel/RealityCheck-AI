from flask_socketio import emit, join_room, leave_room
from app import socketio


@socketio.on('connect', namespace='/ws')
def handle_connect():
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect', namespace='/ws')
def handle_disconnect():
    pass


@socketio.on('subscribe', namespace='/ws')
def handle_subscribe(data):
    analysis_id = data.get('analysis_id')
    if analysis_id:
        join_room(analysis_id)
        emit('subscribed', {'analysis_id': analysis_id})


@socketio.on('unsubscribe', namespace='/ws')
def handle_unsubscribe(data):
    analysis_id = data.get('analysis_id')
    if analysis_id:
        leave_room(analysis_id)
        emit('unsubscribed', {'analysis_id': analysis_id})


def emit_progress(analysis_id: str, stage: str, progress: int, message: str):
    socketio.emit('analysis_progress', {
        'analysis_id': analysis_id,
        'stage': stage,
        'progress': progress,
        'message': message
    }, namespace='/ws', room=analysis_id)
