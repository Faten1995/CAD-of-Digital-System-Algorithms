
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import time
import random
import math
import queue as qu

start_time = time.time()

#Vertix class
class Vertex:
    # id, edges, partition_label
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.goodness = 0

    def add_edge(self, edge):
        # undirected graph, ignore reverse direction
        for present_edge in self.edges:
            if present_edge.left_id == edge.right_id and present_edge.right_id == edge.left_id:
                return
        self.edges.append(edge)

    def edgeCount(self):

        return len(self.edges)


#Edge class
class Edge:
    # left_id, right_id, left_v, right_v
    def __init__(self, left_id, right_id, cost):
        self.left_id = left_id
        self.right_id = right_id
        self.cost = cost

#Graph class
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
                cost += edge.cost
        return cost


#Tabu Class
class Tabu:
    #For the final diagram
    costs =  []
    iteration = []
    avgGoodness =  []
    selectionSet =  []
    tabu_moves_count =  0
    non_tabu_moves_count =  0
    aspiration_moves_count =  0

    def __init__(self, s0, iterations, l1, l2):
        self.graph = s0 #initial solution
        self.iterations = iterations #total iterations for Tabu
        self.tabuListSize = l1 #tabu list size
        self.candidateListSize = l2 #candidate list size
        self.tabuQ = []

        print("Hybird Optimization Function based on Short-memory Tabu search and Simulated Evolution")
        print("Developed by: Faten Adel 2021, fatena42@gmail.com")
        print("Heuristic Parameters:")
        print("Iterations count: " + str(self.iterations))
        print("Tabu List size: " + str(self.tabuListSize))
        print("Candidate List size: " + str(self.candidateListSize))

    def search(self):

        #start with intializing feasible solution
        #initial partitioning
        #Odd numbers in Partition A
        #Even numbers in Partition B
        for i in range(len(self.graph.vertices)):
            if self.graph.vertices[i].id%2 == 0:
                self.graph.vertices[i].partition_label = "B"
            else:
                self.graph.vertices[i].partition_label = "A"

        #print initial solution
        print ("------------Initial Partition---------------")
        print ("V \t Partition")
        for i in range(len(self.graph.vertices)):
            print (str(self.graph.vertices[i].id) +  "\t" + str(self.graph.vertices[i].partition_label))


        #initialize statitics and graph properties.

        self.costs.append(self.graph.get_partition_cost())
        self.iteration.append(0)
        p  = 0


        #start search loop
        for _ in range(self.iterations):
            #call Goodness  function
            #DELETE goodness_array = []
            partition_A = []
            partition_B = []
            total_goodness = 0
            for i in range(len(self.graph.vertices)):
                #DELETE goodness_array.append([self.graph.vertices[i], self.goodnessFunction(self.graph.vertices[i])])
                self.graph.vertices[i].goodness = self.goodnessFunction(self.graph.vertices[i])
                total_goodness = total_goodness + self.graph.vertices[i].goodness
                if self.graph.vertices[i].partition_label == "A":
                    partition_A.append([self.graph.vertices[i], self.graph.vertices[i].goodness])
                else:
                    partition_B.append([self.graph.vertices[i], self.graph.vertices[i].goodness])

            avg = total_goodness / len(self.graph.vertices)
            self.avgGoodness.append(avg)

            #sort based on goodness
            sorted_partition_A = sorted(partition_A, key=lambda x: x[1])
            sorted_partition_B = sorted(partition_B, key=lambda x: x[1])
            potential_solutions = []
            for a in range(int(self.candidateListSize/2)):
                if len(sorted_partition_A) < self.candidateListSize:
                    print("ERROR: Candidate List is Larger than partition size")
                tmpA = sorted_partition_A[a]
                nodeA = tmpA[0]
                #add condition to ensure nodes with good goodness are not included
                if nodeA.goodness > 1:
                    break
                for b in range(int(self.candidateListSize/2)):
                    tmpB = sorted_partition_B[b]
                    nodeB = tmpB[0]
                    #add condition to ensure nodes with good goodness are not included
                    if nodeB.goodness > 1:
                        break
                    cost = self.cost_check(nodeA, nodeB)
                    potential_solutions.append([nodeA, nodeB, cost])

            print("array size " + str(len(potential_solutions)))

            #sort goodness array
            #the  worst goodness is in index 0, the best in index length-1
            sorted_potential_solutions = sorted(potential_solutions, key=lambda x: x[2])


            #print(sorted_goodness_array[0])
            #rint(sorted_goodness_array[self.candidateListSize - 1])

            #initialize and populate the candidate list
            candidate_array = []
            for i in range(self.candidateListSize):
                if i == len(potential_solutions):
                    break
                candidate_array.append(sorted_potential_solutions[i])

            self.selectionSet.append(len(candidate_array))
            for i in range(len(candidate_array)):
                current_solution = candidate_array[i]
                nodeA = current_solution[0]
                nodeB = current_solution[1]
                cost = current_solution[2]

                tabuCheck = self.isTabu(nodeA, nodeB)
                #check the swap is tabu
                #TODO: use Aspiration Critera
                if tabuCheck:
                    self.tabu_moves_count = 1  +   self.tabu_moves_count
                    # print("A: " + str(nodeA.id) + " B:" +str(nodeB.id) + " TABUE MOVE HERE")
                    if self.aspirationOverride(nodeA, nodeB, cost):
                         self.aspiration_moves_count = 1  +   self.aspiration_moves_count
                         # print("Aspiration Override")

                         #SWAP The nodes
                         self.swapper(nodeA, nodeB)

                         #ADD to the Tabu LIST
                         #Remove 100 from the cost to avoid reswapping many times
                         cost = self.graph.get_partition_cost() - 100
                         #maximum tabu length size
                         #Append A to Tabu
                         if len(self.tabuQ) ==  self.tabuListSize:
                             # print("max size, value will be popped: " + str(self.tabuQ[0]))
                             self.tabuQ.pop(0)
                             self.tabuQ.append([nodeA, cost])
                         else:
                             self.tabuQ.append([nodeA, cost])
                         #Append B to Tabu
                         if len(self.tabuQ) ==  self.tabuListSize:
                             # print("max size, value will be popped: " + str(self.tabuQ[0]))
                             self.tabuQ.pop(0)
                             self.tabuQ.append([nodeB, cost])
                         else:
                             self.tabuQ.append([nodeB, cost])
                         break;
                #NOT tabu move, lets swap
                else:
                    self.non_tabu_moves_count = 1  +   self.non_tabu_moves_count
                    self.swapper(nodeA, nodeB)
                    #TODO - add in tabu list
                    cost = self.graph.get_partition_cost()
                    #maximum tabu length size
                    #Append A to Tabu
                    if len(self.tabuQ) ==  self.tabuListSize:
                        # print("max size, value will be popped: " + str(self.tabuQ[0]))
                        self.tabuQ.pop(0)
                        self.tabuQ.append([nodeA, cost])
                    else:
                        self.tabuQ.append([nodeA, cost])
                    #Append B to Tabu
                    if len(self.tabuQ) ==  self.tabuListSize:
                        # print("max size, value will be popped: " + str(self.tabuQ[0]))
                        self.tabuQ.pop(0)
                        self.tabuQ.append([nodeB, cost])
                    else:
                        self.tabuQ.append([nodeB, cost])

                    #break the loop, no need to check more nodes
                    break;


            p = p + 1
            print(str(self.graph.get_partition_cost()))
            self.costs.append(self.graph.get_partition_cost())
            self.iteration.append(p)



        index = self.costs.index(min(self.costs))
        #Print the Final Partition
        print ("------------Final Partition---------------")
        print ("V \t Partition")
        for i in range(len(self.graph.vertices)):
            print (str(self.graph.vertices[i].id) +  "\t" + str(self.graph.vertices[i].partition_label))
        print("The lowest cost is = " + str(min(self.costs)) + " which occured during pass number " +  str(self.iteration[index]))
        print("Tabu moves frequency is " + str(self.tabu_moves_count/self.non_tabu_moves_count))
        print("Non Tabu moves identified is " + str(self.non_tabu_moves_count))
        print("Tabu moves identified is " + str(self.tabu_moves_count))
        print("Tabu moves accepted using the Aspiration criteria is " + str(self.aspiration_moves_count))


        print("--- %s seconds ---" % (time.time() - start_time))

        #Plot cost vs iteration
        x = np.array(self.iteration)
        y = np.array(self.costs) # x is a numpy array now
        plt.plot(x, y)
        # naming the x axis
        plt.xlabel('x - iterations')
        # naming the y axis
        plt.ylabel('y - costs')
        # giving a title to my graph
        plt.title('Cost vs Iteration!')
        plt.show()

        #Plot avg Goodness  vs iteration
        self.iteration.pop(0)
        x = np.array(self.iteration)
        y = np.array(self.avgGoodness) # x is a numpy array now
        plt.plot(x, y)
        # naming the x axis
        plt.xlabel('x - iterations')
        # naming the y axis
        plt.ylabel('y - avg goodness')
        # giving a title to my graph
        plt.title('Average Goodness vs Iteration!')
        plt.show()

        #Plot selection set vs iteration
        x = np.array(self.iteration)
        y = np.array(self.selectionSet) # x is a numpy array now
        plt.plot(x, y)
        # naming the x axis
        plt.xlabel('x - iterations')
        # naming the y axis
        plt.ylabel('y - selection set')
        # giving a title to my graph
        plt.title('Selection Set vs Iteration!')
        plt.show()

        return


    def goodnessFunction(self, v):

        #get the degree
        d = 0
        c = v.edgeCount()
        for i in range(c):
            d = d + v.edges[i].cost
        #check in which partition
        partition = v.partition_label

        #external edges w
        w = 0
        for i in range(c):
            if(v.edges[i].left_id == v.id):
                # this is the same node, now check the other vertix
                if(v.edges[i].right_v.partition_label !=  partition):
                    #print("not a match", partition, v.edges[i].right_v.partition_label)
                    #print("left = ", str(v.edges[i].left_id), " right = ", str(v.edges[i].right_id))
                    w = w + v.edges[i].cost
            elif(v.edges[i].left_v.partition_label !=  partition):
                #print("not a match", partition, v.edges[i].left_v.partition_label)
                #print("left = ", str(v.edges[i].left_id), " right = ", str(v.edges[i].right_id))
                w = w + v.edges[i].cost

        # 1 - (w/d)

        g = 1 - (w/d)

        #print(g)
        return g

#Neighbor function to CHECK COST
    def cost_check(self, nodeA, nodeB):

        #Swap node A and B
        for v in self.graph.vertices:
            if v.id == nodeA.id:
                v.partition_label = "B"
            elif v.id == nodeB.id:
                v.partition_label = "A"

        new_cost = self.graph.get_partition_cost()

        #Reswap again
        for v in self.graph.vertices:
            if v.id == nodeA.id:
                v.partition_label = "A"
            elif v.id == nodeB.id:
                v.partition_label = "B"

        return new_cost

        #TODO - define aspiration criteria
    def aspirationOverride(self, nodeA, nodeB, cost):
        if len(self.tabuQ) == 0 :
            return 0
        for i in range(len(self.tabuQ)):
            tmp  = self.tabuQ[i]
            tmpNode = tmp[0]
            tmpCost = tmp[1]
            #tmpB = tmp[1]
            #print(str(tmpA.id) + " == " + str(nodeA.id) + " and " + str(tmpB.id) + " == " +str(nodeB.id))
            if tmpNode.id == nodeA.id or tmpNode.id == nodeB.id:
                if tmpCost > cost:
                    # print("new cost: " + str(cost) + " old cost: " + str(tmpCost))
                    self.tabuQ.pop(i)
                    return 1
        return 0

        #TODO - check if swap is in tabu list
    def isTabu(self, nodeA, nodeB):
        if len(self.tabuQ) == 0 :
            return 0
        for i in range(len(self.tabuQ)):
            tmp  = self.tabuQ[i]
            tmpNode = tmp[0]
            if tmpNode.id == nodeA.id or tmpNode.id == nodeB.id:
                # print("TABOOOOOOO")
                return 1

        return 0
    def swapper(self, nodeA, nodeB):

        #Swap node A and B
        for v in self.graph.vertices:
            if v.id == nodeA.id:
                v.partition_label = "B"
            elif v.id == nodeB.id:
                v.partition_label = "A"

        return
#Main function
def main():
    graph = load_data()
    #kl = KernighanLin(graph)
    #kl.partition()

    #graph, iterations count, tabu list size, candidate list size
    #if Tmp is 0 and alpa is 1 it will be a greedy function
    sa = Tabu(graph, 50, 30, 30)
    sa.search()

#Function to load data
def load_data():
    #file = open(filename, 'r')
    file = open("/Users/faten/Documents/Tabu and SimE/10by10withCost_2.txt")

    edges = []
    vertices = []
    seen_vertex_ids = []

    for line in list(file):
        v_list = line.split()
        left_id = int(v_list[0])
        right_id = int(v_list[1])
        cost_id = int(v_list[2])
        edges.append(Edge(left_id, right_id, cost_id))

        if left_id not in seen_vertex_ids:
            vertices.append(Vertex(left_id))
            seen_vertex_ids.append(left_id)

        if right_id not in seen_vertex_ids:
            vertices.append(Vertex(right_id))
            seen_vertex_ids.append(right_id)

    return Graph(vertices, edges)

#Start the application
if __name__ == "__main__":
    main()
