# Implementation of Kernighan-Lin graph partitioning algorithm

#This code is a modified version from the following paper: An Efficient Heuristic Procedure for Partitioning Graphs (https://ieeexplore.ieee.org/document/6771089)
#Modifications include
    # 1- Understood and modified the source code from Python2 to Python3
    # 2- Changed the greediness of the algorithm to ensure that the KL algorithms loops through all iterations until all nodes are swapped
    # 3- Included the swapped fixed D_values
    # 4- Included the plotting of the iterations vs the costs
    # 5-  Removed the c calculations to  test the speed and the differences in the results
    # 6- Included lists to identify the loswet cost occured during which iteration
    # 7- Even and Odd Partitioning

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import time
start_time = time.time()


class Vertex:
    # id, edges, partition_label
    def __init__(self, id):
        self.id = id
        self.edges = []
        #fixed  is used to indicate if this vertex has been moved, the value should be changed to 1 to avoid re-swapping again.
        #The loop should end if there are no more vertexes with fixed = 0
        self.fixed = 0
        # 1 even, 0 odd: initially all nodes are initiated as even.
        self.isEven = 1


    def set_v_fixed(slef):
        print("inside set v fixed")
        self.fixed = 1
        return

    def get_D_value(self):
        D_value = 0 # D = E - I

        for edge in self.edges:
            if edge.left_id == self.id:
                other_v = edge.right_v
            elif edge.right_id == self.id:
                other_v = edge.left_v

            if other_v.partition_label != self.partition_label:
                D_value += 1 # external cost
            # else:
            #     D_value -= 1 # internal cost

        return D_value

    def add_edge(self, edge):
        # undirected graph, ignore reverse direction
        for present_edge in self.edges:
            if present_edge.left_id == edge.right_id and present_edge.right_id == edge.left_id:
                return

        self.edges.append(edge)

class Edge:
    # left_id, right_id, left_v, right_v
    def __init__(self, left_id, right_id):
        self.left_id = left_id
        self.right_id = right_id

class Graph:
    # vertices, edges
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

        # connect vertices and edges
        vertex_dict = {v.id: v for v in self.vertices}

        for edge in self.edges:
            edge.left_v = vertex_dict[edge.left_id]
            vertex_dict[edge.left_id].add_edge(edge)

            edge.right_v = vertex_dict[edge.right_id]
            vertex_dict[edge.right_id].add_edge(edge)

    def get_partition_cost(self):
        cost = 0

        for edge in self.edges:
            if edge.left_v.partition_label != edge.right_v.partition_label:
                cost += 1

        return cost


class KernighanLin():
    costs =  []
    iteration = []
    def __init__(self, graph):
        self.graph = graph

    def partition(self):
        # initial partition: first half is group A, second half is B
        # CHANGED - This section will be commented to make the partitioning based on even/odd as requested.
        # for i in range(len(self.graph.vertices)//2):
        #     self.graph.vertices[i].partition_label = "A"
        # for i in range(len(self.graph.vertices)//2, len(self.graph.vertices)):
        #     self.graph.vertices[i].partition_label = "B"
        # CHANGED - This is the new partitioning logic:
        # loop to identify even/odd nodes
        for i in range(len(self.graph.vertices)):
            if int(i) % 2 == 0:
                self.graph.vertices[i].isEven = 1
                self.graph.vertices[i].partition_label = "A"
            else:
                self.graph.vertices[i].isEven = 0
                self.graph.vertices[i].partition_label = "B"

        self.costs.append(self.graph.get_partition_cost())
        self.iteration.append(0)
        print ("Initial partition cost: " + str(self.graph.get_partition_cost()))
        p = 0 # pass
        total_gain = 0
        print ("Initial partition elements: ")
        print ("------------A---------------")
        print ("V \t Fixed \t Partition")
        for i in range(len(self.graph.vertices)):
            if self.graph.vertices[i].partition_label == "A":
                print (str(self.graph.vertices[i].id) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label))
        print ("------------B---------------")
        print ("V \t Fixed \t Partition")
        for i in range(len(self.graph.vertices)):
            if self.graph.vertices[i].partition_label == "B":
                print (str(self.graph.vertices[i].id) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label))

        # repeat until g_max <= 0
        resume = 1
        while True:
            group_a = []
            group_b = []


            #self.graph.vertices[i].fixed != 1
            for i in range(len(self.graph.vertices)):
                if self.graph.vertices[i].partition_label == "A" and self.graph.vertices[i].fixed != 1:
                    group_a.append(self.graph.vertices[i])
                elif self.graph.vertices[i].partition_label == "B" and self.graph.vertices[i].fixed != 1:
                    group_b.append(self.graph.vertices[i])

            D_values = {v.id: v.get_D_value() for v in self.graph.vertices}
            gains = [] # [ ([a, b], gain), ... ]

            # while there are unvisited vertices
            for _ in range(len(self.graph.vertices)):
                D_values = {v.id: v.get_D_value() for v in self.graph.vertices}
                total_gain = 0
                # choose a pair that maximizes gain
                max_gain = -1 * float("inf") # -infinity
                #max_gain = -1 * int("inf") # -infinity
                pair = []

                for a in group_a:
                    for b in group_b:
                        c_ab = len(set(a.edges).intersection(b.edges))
                        # to test part b
                        gain = D_values[a.id] + D_values[b.id] - (2 * c_ab)
                        # gain = D_values[a.id] + D_values[b.id]
                        if gain > max_gain:
                            max_gain = gain
                            pair = [a, b]



                # mark that pair as visited
                # a = pair[1]
                # b = pair[1]
                #group_a.remove(varA.id)
                #group_b.remove(varB.id)
                gains.append([[a, b], max_gain])
                #





            # find j that maximizes the sum g_max
            g_max = -1 * float("inf")
            #print("inital g_max value =", g_max)
            #g_max = -1 * int("inf")

            jmax = 0
            for j in range(1, len(gains) + 1):
                g_sum = 0
                # print("gains")
                for i in range(j):
                    g_sum += gains[i][1]
                    # print(gains[i][1])
                # print("g_sum = " + str(g_sum) +" g_max = " + str(g_max))
                if g_sum > g_max:
                    g_max = g_sum
                    jmax = j


            if g_max > 0 and resume:
            #if self.graph.get_partition_cost() != 10:
                # swap in graph
                # #FATEN - There must  be a  way to fix the vertices, to avoid re-swapping them.
                # #FATEN - Modify the algorith and try to append a lable for fixed nodes
                nodeA = gains[i][0][0].id
                nodeB = gains[i][0][1].id
                print("swapped nodes are " + str(nodeA) + " " + str(nodeB))
                for i in range(jmax):
                    # find vertices and change their partition_label
                    for v in self.graph.vertices:
                        if v.id == gains[i][0][0].id:
                            v.partition_label = "B"
                            v.fixed = 1
                        elif v.id == gains[i][0][1].id:
                            v.partition_label = "A"
                            v.fixed = 1
                # update D_values of other unvisited nodes connected to a and b, as if a and b are swapped
                for x in group_a:
                    if x != nodeA:
                        c_xa = len(set(x.edges).intersection(a.edges))
                        c_xb = len(set(x.edges).intersection(b.edges))
                        D_values[x.id] += 2 * (c_xa) - 2 * (c_xb)
                #     #print("c_xa", c_xa)
                #     #print("c_xb", c_xb)
                    # print("D_values[x.id]", D_values[x.id])

                for y in group_b:
                    if y != nodeB:
                        c_yb = len(set(y.edges).intersection(b.edges))
                        c_ya = len(set(y.edges).intersection(a.edges))
                        D_values[y.id] += 2 * (c_yb) - 2 * (c_ya)

                p += 1
                # print ("Parition Elements of Pass: ", p)
                # print ("V \t Fixed \t Partition")
                # for i in range(len(self.graph.vertices)):
                #     if self.graph.vertices[i].partition_label == "A":
                #         print (str(i) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label))
                # for i in range(len(self.graph.vertices)):
                #     if self.graph.vertices[i].partition_label == "B":
                #         print (str(i) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label))

                total_gain += g_max
                print ("Pass: " + str(p)+  "\t\tFinal partition cost: " + str(self.graph.get_partition_cost()))
                print ("------------A---------------")
                print ("V \t Fixed \t Partition \t D Value")
                for i in range(len(self.graph.vertices)):
                    if self.graph.vertices[i].partition_label == "A":
                        print (str(self.graph.vertices[i].id) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label) + "\t" + str(self.graph.vertices[i].get_D_value()))
                print ("------------B---------------")
                print ("V \t Fixed \t Partition \t D Value")
                for i in range(len(self.graph.vertices)):
                    if self.graph.vertices[i].partition_label == "B":
                        print (str(self.graph.vertices[i].id) + "\t" + str(self.graph.vertices[i].fixed) + "\t" + str(self.graph.vertices[i].partition_label) + "\t" + str(self.graph.vertices[i].get_D_value()))

                if self.graph.get_partition_cost() > max(self.costs):
                    resume = 0
                else:
                    self.costs.append(self.graph.get_partition_cost())
                    self.iteration.append(p)

            else:

                #print(self.costs)
                #print(self.iteration)

                #plotting the graphs
                x = np.array(self.iteration)
                y = np.array(self.costs) # x is a numpy array now
                plt.plot(x, y)

                # naming the x axis
                plt.xlabel('x - iterations')
                # naming the y axis
                plt.ylabel('y - costs')

                # giving a title to my graph
                plt.title('Cost vs Iteration!')

                #print the lowest cost
                #find the index of the minmum cost
                index = self.costs.index(min(self.costs))
                print("The lowest cost is = " + str(min(self.costs)) + " which occured during pass number " +  str(self.iteration[index]))
                # function to show the plot

                total_gain += g_max
                print ("Fail: " + str(p) + "\t\t\tLoss: " + "\t\tFinal partition cost: " + str(self.graph.get_partition_cost()))

                break

        print ("Total passes: " + str(p) + "\t\tFinal partition cost: " + str(self.graph.get_partition_cost()))
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.show()

def main():
    graph = load_data("/Users/faten/Documents/KL/10by10.txt")
    kl = KernighanLin(graph)
    kl.partition()

def load_data(filename):
    #file = open(filename, 'r')
    file = open("/Users/faten/Documents/KL/10by10.txt")

    edges = []
    vertices = []
    seen_vertex_ids = []

    for line in list(file):
        v_list = line.split()
        left_id = int(v_list[0])
        right_id = int(v_list[1])

        edges.append(Edge(left_id, right_id))

        if left_id not in seen_vertex_ids:
            vertices.append(Vertex(left_id))
            seen_vertex_ids.append(left_id)

        if right_id not in seen_vertex_ids:
            vertices.append(Vertex(right_id))
            seen_vertex_ids.append(right_id)

    return Graph(vertices, edges)

if __name__ == "__main__":
    main()
