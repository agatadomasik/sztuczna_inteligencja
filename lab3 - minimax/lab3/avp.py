from exceptions import GameplayException
from connect4 import Connect4
from minmaxagent import MinMaxAgent

connect4 = Connect4(width=6, height=7)
agent = MinMaxAgent('x')
while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent.my_token:
            n_column = agent.decide(connect4)
        else:
            n_column = int(input(':'))
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
