
from scipy import stats
import numpy as np
import time
import random
import math
from typing import List, Optional, Callable, Tuple
from random import choices, randint, randrange, random
import pytest
import bisect
import logging
import itertools

class Node:
    def __init__(self, value, hw1, hw2, r):
        self.value = value
        #height and width 1
        self.hw1 = hw1
        #height and width 2
        self.hw2 = hw2
        #Rotatable (1 is True), default is false
        self.r = r
        self.minimum = []

    def __str__( self):
        return str(self.value)

class Tree:
    def __init__(self, l, r, root):
        self.l = l
        self.r = r
        #operand
        self.root = root
        self.minimum = []

    def __str__( self):
        return "(" + str(self.l) + self.root.value + str(self.r) + ")"

def build_tree(polish):
    stack = []
    for obj in polish:
        #print(str(obj))
        if obj.value in ["|","-"]:
            r = stack.pop()
            l = stack.pop()
            stack.append(Tree( l, r, obj))
        else:
            stack.append(obj)

    assert len(stack) == 1

    return stack[0]


# Postorder traversal
# Left ->Right -> Root
def PostorderTraversal(tree):
    res = []
    if tree:
        if isinstance(tree, Tree):
            res = PostorderTraversal(tree.l)
            res = res + PostorderTraversal(tree.r)
            res.append(tree.root.value)
        else:
            res.append(tree.value)
    return res


def calculateMinimum(tree):
    res = []
    if tree:
        if isinstance(tree, Tree):
            calculateMinimum(tree.l)
            calculateMinimum(tree.r)
        else:

            #check if the boxes are Rotatable
            if tree.r == 1:
                item1 = tree.hw1
                item2 = tree.hw2
                item3 = [tree.hw1[1], tree.hw1[0]]
                item4 = [tree.hw2[1], tree.hw2[0]]

                #list of heights and widths
                hw_list = [item1, item2, item3, item4]
                print("original " + str(hw_list))
                #remove duplicates
                hw_list.sort()
                hw_list = list(hw_list for hw_list,_ in itertools.groupby(hw_list))
                print("remove duplicates " + str(hw_list))

                # Minimum using first element
                hw1 = min(hw_list, key=lambda x: x[0])
                # Minimum using first element
                hw2 = min(hw_list, key=lambda x: x[1])

                print("hw1: " + str(hw1) + " hw2: " + str(hw2))

                minimum = []

                # hw1 = tree.hw1
                # hw2 = tree.hw2

                print(str(hw1) + "\t"+ str(hw2))
                if hw1[0] < hw2[0]:
                    # if hw1[0] is smaller, and hw1[1] is equal or smaller append hw1
                    if hw1[1] <= hw2[1]:
                        minimum.append(hw1)
                    else:
                        minimum.append(hw1)
                        minimum.append(hw2)
                elif hw2[0] < hw1[0]:
                    if hw2[1] <= hw1[1]:
                        minimum.append(hw2)
                    else:
                        minimum.append(hw1)
                        minimum.append(hw2)
                else:
                    minimum.append(hw1)
                    minimum.append(hw2)

                tree.minimum = minimum
                print(str(tree.value) + " minimum: " + str(minimum))
            else:
                minimum = []
                hw1 = tree.hw1
                hw2 = tree.hw2
                print(str(hw1) + "\t"+ str(hw2))
                if hw1[0] < hw2[0]:
                    # if hw1[0] is smaller, and hw1[1] is equal or smaller append hw1
                    if hw1[1] <= hw2[1]:
                        minimum.append(hw1)
                    else:
                        minimum.append(hw1)
                        minimum.append(hw2)
                elif hw2[0] < hw1[0]:
                    if hw2[1] <= hw1[1]:
                        minimum.append(hw2)
                    else:
                        minimum.append(hw1)
                        minimum.append(hw2)
                else:
                    minimum.append(hw1)
                    minimum.append(hw2)

                tree.minimum = minimum
                print(str(tree.value) + " minimum: " + str(minimum))
    return

def findBoundingBox(tree):
    res = []
    if tree:
        if isinstance(tree, Tree):
            findBoundingBox(tree.l)
            findBoundingBox(tree.r)
            if tree.root.value == "|":
                print("value is |")
                print("L: " + str(tree.l))
                print("R: " + str(tree.r))
                print("L minimum: " + str(tree.l.minimum))
                print("R minimum: " + str(tree.r.minimum))

                valuel0 = tree.l.minimum[0]
                valuel1 = tree.l.minimum[1]
                valuer0 = tree.r.minimum[0]
                valuer1 = tree.r.minimum[1]
                #first attempt
                w = valuel0[0] + valuer0[0]
                h = max(valuel0[1], valuer0[1])
                tmp1 =  [w, h]
                w = valuel0[0] + valuer1[0]
                h = max(valuel0[1], valuer1[1])
                tmp2 =  [w, h]
                w = valuel1[0] + valuer1[0]
                h = max(valuel1[1], valuer1[1])
                tmp3 =  [w, h]
                w = valuel1[0] + valuer0[0]
                h = max(valuel1[1], valuer0[1])
                tmp4 =  [w, h]

                #create list
                tmp_list = [tmp1, tmp2, tmp3, tmp4]
                tmp_list.sort()
                #remove duplicates
                tmp_list = list(tmp_list for tmp_list,_ in itertools.groupby(tmp_list))

                # Minimum using first element
                min1 = min(tmp_list, key=lambda x: x[0])
                # Minimum using first element
                min2 = min(tmp_list, key=lambda x: x[1])

                tree.minimum = [min1, min2]

                print(tree.minimum)
            elif tree.root.value == "-":
                print("value is -")
                print("L: " + str(tree.l))
                print("R: " + str(tree.r))
                print("L minimum: " + str(tree.l.minimum))
                print("R minimum: " + str(tree.r.minimum))

                valuel0 = tree.l.minimum[0]
                valuel1 = tree.l.minimum[1]
                valuer0 = tree.r.minimum[0]
                valuer1 = tree.r.minimum[1]
                #first attempt
                w = max(valuel0[0], valuer0[0])
                h = valuel0[1] + valuer0[1]
                tmp1 =  [w, h]
                w = max(valuel0[0], valuer1[0])
                h = valuel0[1] + valuer1[1]
                tmp2 =  [w, h]
                w = max(valuel1[0], valuer1[0])
                h = valuel1[1] + valuer1[1]
                tmp3 =  [w, h]
                w = max(valuel1[0], valuer0[0])
                h = valuel1[1] + valuer0[1]
                tmp4 =  [w, h]

                #create list
                tmp_list = [tmp1, tmp2, tmp3, tmp4]
                tmp_list.sort()
                #remove duplicates
                tmp_list = list(tmp_list for tmp_list,_ in itertools.groupby(tmp_list))

                # Minimum using first element
                min1 = min(tmp_list, key=lambda x: x[0])
                # Minimum using first element
                min2 = min(tmp_list, key=lambda x: x[1])

                tree.minimum = [min1, min2]

                print(tree.minimum)
            else:
                print("ERROR")
    return tree.minimum

#Main function
def main():
    #graph = load_data("/Users/faten/Documents/Floor_Planning/test.txt")


    nodex = Node("-", [0,0], [0,0], 0)
    nodey = Node("|", [0,0], [0,0], 0)
    node1 = Node(1, [32,4], [16,8],  1)
    node2 = Node(2, [32,4], [16,8],  1)
    node3 = Node(3, [32,4], [16,8],  1)
    node4 = Node(4, [32,4], [16,8],  1)
    node5 = Node(5, [32,4], [16,8],  1)
    node6 = Node(6, [32,4], [16,8],  1)
    # polish_expression = [node1, node2, "-", node3, node4, "|", node5, node6, "|", "-", "|"]
    polish_expression = [node1, node2, nodex, node3, node4, nodey, node5, node6, nodey, nodex, nodey]
    tree = build_tree(polish_expression)

    #postorder_parser(tree)
    #print(tree)
    #print(PostorderTraversal(tree))
    calculateMinimum(tree)
    minimum_value = findBoundingBox(tree)

    box1 = minimum_value[0]
    box2 = minimum_value[1]

    box1_size = box1[0] * box1[1]
    box2_size = box2[0] * box2[1]

    if box1_size < box2_size:
        print("Minimum Bounding Box Area Size: " + str(box1_size) + " dimensions: " + str(box1))
    else:
        print("Minimum Bounding Box Area Size: " + str(box2_size) + " dimensions: " + str(box2))

    return

#Start the application
if __name__ == "__main__":
    main()
