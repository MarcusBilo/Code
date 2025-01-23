# https://jamesmccaffrey.wordpress.com/2024/06/26/computing-a-measure-of-the-sharpness-of-a-chess-position-using-stockfish/

from stockfish import Stockfish
import numpy as np

print("\nBegin position sharpness analysis ")

loc = r"D:\Ablage\c old dekstop u dokumente 2024\Ordner\nibbler-2.3.2\stockfish-17-windows-x86-64-bmi2\stockfish-windows-x86-64-bmi2.exe"

stockfish = Stockfish(path=loc, depth=20, parameters={"Threads": 3, "Minimum Thinking Time": 30, "Hash": 2048})

p = stockfish.get_parameters()
print("\nStockfish engine parameters: ")
print(p)




pos_fen = "rnbqkb1r/pp2pppp/8/2pn4/8/5NP1/PP1PPP1P/RNBQKB1R w KQkq - 0 5"
print("\nPosition in FEN: ")
print(pos_fen)



stockfish.set_fen_position(pos_fen)

vis_pos = stockfish.get_board_visual()
print("\nPosition visual: ")
print(vis_pos)

curr_eval = stockfish.get_evaluation()
print("\nPosition evaluation = ", end="")
print(curr_eval)
curr_val = curr_eval["value"]

bms = stockfish.get_top_moves(5)
print("\nBest 5 moves in this position:")
for i in range(len(bms)):
    print(bms[i])  # could be less than 5

# this code doesn't take into account a forced mate
#  when ["Centipawn"] in None. See below.

print("\nComputing sharpness ")
sum = 0
for i in range(len(bms)):
    poss_val = bms[i]["Centipawn"]
    delta = np.abs(curr_val - poss_val)
    if (curr_val > 0 and poss_val < 0) or \
            (curr_val < 0 and poss_val > 0):
        delta *= 2
    print("delta " + str(i) + " = " + str(delta))
    sum += delta

sharpness = sum / len(bms)  # could be less than 5
print("\nPosition sharpness = %0.1f " % sharpness)


# compute sharpness using a function with no chit-chat
# deal with foced mate positions

def compute_sharpness(pos_fen):
    stockfish.set_fen_position(pos_fen)
    curr_eval = stockfish.get_evaluation()
    if curr_eval is None: return 0.0
    curr_val = curr_eval["value"]

    bms = stockfish.get_top_moves(5)
    if bms is None or len(bms) == 0:
        return 0.0

    sum = 0
    for i in range(len(bms)):  # could be less than 5
        if bms[i]["Mate"] is not None:  # forced mate
            # print("a forced mate exists")
            if bms[i]["Mate"] < 0:  # black mates white
                poss_val = -400
            elif bms[i]["Mate"] > 0:  # white mates black
                poss_val = 400
        else:
            poss_val = bms[i]["Centipawn"]

        delta = np.abs(curr_val - poss_val)
        if (curr_val > 0 and poss_val < 0) or (curr_val < 0 and poss_val > 0):
            delta *= 2
        sum += delta
    return sum / len(bms)


# pos_fen = "r2qr3/pp1b1pkp/2ppnn2/4pp2/3PP3/" + \
#   "P3PQNP/BPP3P1/R4RK1 w - - 0 21"
# pos_sharpness = compute_sharpness(pos_fen)
# print("\nPosition sharpness = %0.1f " % pos_sharpness)

print("\nEnd analysis ")
