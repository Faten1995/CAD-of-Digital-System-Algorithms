
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import time
import random
import math
start_time = time.time()

#Vertix class
class Vertex:
    # id, edges, partition_label
    def __init__(self, id):
        self.id = id
        self.edges = []

    def add_edge(self, edge):
        # undirected graph, ignore reverse direction
        for present_edge in self.edges:
            if present_edge.left_id == edge.right_id and present_edge.right_id == edge.left_id:
                return
        self.edges.append(edge)

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

#Simulated Annealing class
class Simulated_Annealing:
    #For the final diagram
    costs =  []
    iteration = []
    def __init__(self, s0, temp0, alpha, beta, m, max_time):
        self.graph = s0 #initial solution
        self.temp0 = temp0 #initial temprature
        self.alpha = alpha #the cooling rate
        self.beta = beta #constant
        self.M = m #time until next parameter update
        self.max_time = max_time #total time allowed for annealing process

        print("Simulated Annealing Heuristic with Weights")
        print("Heuristic Parameters:")
        print("Intial Temprature (temp0): " + str(self.temp0))
        print("Cooling Rate (alpha): " + str(self.alpha))
        print("Beta constant: " + str(self.beta))
        print("Time until next parameter update (M): " + str(self.M))
        print("Max Time: " + str(self.max_time))

#Annealing Seems OK
    def annealing(self):

        #initial partitioning
        for i in range(len(self.graph.vertices)//2):
            self.graph.vertices[i].partition_label = "A"
        for i in range(len(self.graph.vertices)//2, len(self.graph.vertices)):
            self.graph.vertices[i].partition_label = "B"

        # print ("------------Initial Partition---------------")
        # print ("V \t Partition")
        # for i in range(len(self.graph.vertices)):
        #     print (str(self.graph.vertices[i].id) +  "\t" + str(self.graph.vertices[i].partition_label))

        #self.costs.append(self.graph.get_partition_cost())
        #self.iteration.append(0)
        #print ("Initial partition cost: " + str(self.graph.get_partition_cost()))
        self.costs.append(self.graph.get_partition_cost())
        self.iteration.append(0)
        temp = self.temp0
        S = self.graph
        run_time = 0
        p  = 1
        M = self.M
        while(run_time <= self.max_time):
            self.metropolis(S, temp, M)
            print("iteration: " + str(p) + " cost: " + str(self.graph.get_partition_cost()))
            self.costs.append(self.graph.get_partition_cost())
            self.iteration.append(p)
            p += 1
            run_time = run_time + M
            #reducing the temprature
            temp = self.alpha * temp
            M = self.beta * self.M
            if(temp<=0):
                break



        index = self.costs.index(min(self.costs))
        print("The lowest cost is = " + str(min(self.costs)) + " which occured during pass number " +  str(self.iteration[index]))
        #Print the Final Partition
        # print ("------------Final Partition---------------")
        # print ("V \t Partition")
        # for i in range(len(self.graph.vertices)):
        #     print (str(self.graph.vertices[i].id) +  "\t" + str(self.graph.vertices[i].partition_label))


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
        return

#Metropolis function NOT DONE
    #def metropolis(self, S, temp, M):
    def metropolis(self, S, temp, M):
        m = int(M)
        while( m > 0):
            #Generate random number between 0 and 1
            randomX= random.uniform(0,1)
            current_cost = self.graph.get_partition_cost()
            new_cost, nodeA, nodeB = self.neighbor_check()
            #print(str(current_cost) + "\n" + str(new_cost) + "\n")
            delta_h = new_cost - current_cost
            value = delta_h / temp
            try:
                exponent = math.exp(-value)
                #print("randomX = " + str(randomX) + "exponent = " + str(exponent))

            except OverflowError:
                exponent = 0
            if((delta_h < 0) or (randomX<exponent)):
                #SWAP
                self.neighbor(nodeA, nodeB)
            m = m - 1
        return

#Neighbor function to CHECK COST
    def neighbor_check(self):
        group_a = []
        group_b = []
        #Fill the partitions arrays
        for i in range(len(self.graph.vertices)):
            if self.graph.vertices[i].partition_label == "A":
                group_a.append(self.graph.vertices[i])
            elif self.graph.vertices[i].partition_label == "B":
                group_b.append(self.graph.vertices[i])
        #array size
        a_size = len(group_a)
        b_size = len(group_b)
        #Generate random values of elements to be swapped
        randomA =  random.randint(1,a_size) - 1
        randomB =  random.randint(1,b_size) - 1
        nodeA = group_a[randomA].id
        nodeB = group_b[randomB].id

        #Swap node A and B
        for v in self.graph.vertices:
            if v.id == nodeA:
                v.partition_label = "B"
            elif v.id == nodeB:
                v.partition_label = "A"


        new_cost = self.graph.get_partition_cost()

        #Reswap again
        for v in self.graph.vertices:
            if v.id == nodeA:
                v.partition_label = "A"
            elif v.id == nodeB:
                v.partition_label = "B"

        return new_cost, nodeA, nodeB

#Neighbor function to SWAP
    def neighbor(self, nodeA, nodeB):

        #print("inside swapper")
        group_a = []
        group_b = []
        #Fill the partitions arrays
        for i in range(len(self.graph.vertices)):
            if self.graph.vertices[i].partition_label == "A":
                group_a.append(self.graph.vertices[i])
            elif self.graph.vertices[i].partition_label == "B":
                group_b.append(self.graph.vertices[i])


        #Swap node A and B
        for v in self.graph.vertices:
            if v.id == nodeA:
                v.partition_label = "B"
            elif v.id == nodeB:
                v.partition_label = "A"

        return

#Cost function NOT DONE
    def cost(self):
        #print("inside cost")
        return

#Main function
def main():
    graph = load_data("/Users/faten/Documents/Simulated_Annealing/100by100.txt")
    #kl = KernighanLin(graph)
    #kl.partition()

    #graph, temp, cooling rate, beta constant, M, maxtime
    #if Tmp is 0 and alpa is 1 it will be a greedy function
    sa = Simulated_Annealing(graph, 1000000, 0.99, 0.5, 5, 10000)
    sa.annealing()


#Function to load data
def load_data(filename):
    #file = open(filename, 'r')
    file = open("/Users/faten/Documents/Simulated_Annealing/10by10withCost.txt")

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
