from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from alphabetaagent import AlphaBetaAgent
from minmaxagent import MinMaxAgent

connect4 = Connect4(width=6, height=7)
#agent1 = RandomAgent('o')
agent1 = MinMaxAgent('x')
agent2 = AlphaBetaAgent('o')
while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent1.my_token:
            n_column = agent1.decide(connect4)
        else:
            n_column = agent2.decide(connect4)
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
