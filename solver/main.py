import time

import cv2

from colorArrayResize import colorArrayResize
from contouring import (checkContoursAndDraw, findAngles,
                        find_90_deg_angles, find_hidden_square,
                        segment_contour, applyBinaryAndDrawContours)
from Piece import Piece
from zeb.cutColorBand import cutColorBand
from zeb.colorTest import colorTest

if __name__ == "__main__":
    SIDE_SAMPLES_COUNT = 30
    originalImage = cv2.imread('../images/allpiecesflash/align.png')
    contours = applyBinaryAndDrawContours(originalImage, draw=False)
    pieces = []

    for contour in range(0, len(contours)):
        ravelledContours = checkContoursAndDraw(contours, contour, draw=False)
        useful_points = findAngles(ravelledContours)
        allPermutations = find_90_deg_angles(useful_points)
        hiddenSquare = find_hidden_square(allPermutations)
        segmented_sides = segment_contour(ravelledContours, hiddenSquare, draw=False)

        sides_colors = []
        for side in segmented_sides:
            side_colors = []
            for point in side:
                side_colors.append(originalImage[point[1], point[0]])
            sides_colors.append(cutColorBand(side_colors, SIDE_SAMPLES_COUNT))

        pieces.append(Piece(
            contour,
            sides_colors,
        ))

    print("SOLVER: found ", len(pieces), " pieces")

    sides_comp_memo = {}
    def sides_comp(p1, p2, side1, side2):
        # returns the similarity coefficient between two sides of two pieces

        (side1, side2) = (side1, side2) if p1.id < p2.id else (side2, side1)
        (p1, p2) = (p1, p2) if p1.id < p2.id else (p2, p1)

        if (p1.id, p2.id, side1, side2) in sides_comp_memo:
            return sides_comp_memo[(p1.id, p2.id, side1, side2)]

        comp = colorTest(p1.sides[side1], p2.sides[side2])

        sides_comp_memo[(p1.id, p2.id, side1, side2)] = comp
        return comp

    class PuzzleNode:
        def __init__(self, piece, matchings, used_pieces, score, parent=None):
            self.piece = piece
            self.matchings = matchings
            self.used_pieces = used_pieces
            self.score = score
            self.parent = parent

        def __copy__(self):
            return PuzzleNode(self.piece, self.matchings.copy(), self.used_pieces.copy(), self.score, self.parent)


    end_time = time.time() + 60
    def solve_bfs(pieces):
        # Breadth-first search to solve the puzzle
        # Returns the solution node


        # Initialize the queue with the first piece
        queue = [PuzzleNode(pieces[0], [None] * 4, {0}, 0)]
        best_solution_so_far = queue[0]

        while queue:
            node = queue.pop(0)

            if node.score > best_solution_so_far.score and len(node.used_pieces) > len(best_solution_so_far.used_pieces):
                best_solution_so_far = node

            if time.time() > end_time or len(node.used_pieces) == len(pieces):
                return best_solution_so_far

            matchings = []
            for i, piece in enumerate(pieces):
                if piece.id in node.used_pieces:
                    continue

                for j in range(4):
                    if node.matchings[j] is not None:
                        continue

                    for k in range(4):
                        comp = sides_comp(node.piece, piece, j, k)
                        if comp < 0.7:
                            continue

                        matchings.append((i, j, k, comp))

            # keep only the best 5 matchings
            matchings.sort(key=lambda x: x[3], reverse=True)
            matchings = matchings[:5]

            for i, j, k, comp in matchings:
                new_node = node.__copy__()
                new_node.matchings.append((i, k, node.piece.id, j))
                new_node.piece = pieces[i]
                new_node.used_pieces.add(i)
                new_node.score += comp
                queue.append(new_node)

        return best_solution_so_far

    print("SOLVER: starting solving")
    solution = solve_bfs(pieces)
    print("SOLVER: done solving")
    print(solution)








