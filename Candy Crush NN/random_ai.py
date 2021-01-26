import random
import graphical, game
import numpy as np

def ai_callback(board, score, moves_left):
    dir = random.randint(0, 1) == 0
    return (random.randint(0, 7 if dir else 6), random.randint(0, 8 if dir else 9), dir)

def transition_callback(board, move, score_delta, next_board, moves_left):
    pass # This can be used to monitor outcomes of moves

def end_of_game_callback(boards, scores, moves, final_score):
    return True # True = play another, False = Done


if __name__ == '__main__':
    speedup = 1.0
    g = graphical.Game(ai_callback, transition_callback, end_of_game_callback, speedup)
    g.run()
