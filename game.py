from tensorflow.keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(player, bot):
    rules = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}

    if player == bot:
        return "Tie"
    elif bot == rules[player]:
        return 'User'
    else:
        return 'Bot'


model = load_model("model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    width = 1500
    height = 1080
    dim = (width, height)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    cv2.rectangle(frame, (100, 100), (600, 600), (255, 255, 255), 2)

    cv2.rectangle(frame, (800, 100), (1300, 600), (255, 255, 255), 2)

    roi = frame[100:600, 100:600]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    if prev_move != user_move_name:
        if user_move_name != "none":
            bot_move = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, bot_move)
        else:
            bot_move = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Bot's Move: " + bot_move,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 800), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if bot_move != "none":
        icon = cv2.imread(
            "{}.png".format(bot_move))
        icon = cv2.resize(icon, (500, 500))
        frame[100:600, 800:1300] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
