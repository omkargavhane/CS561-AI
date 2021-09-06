from tiles_inpt import *
import time
from pprint import pprint
import copy

open_list=[]
closed_list=[]
algo=None
dist_metric=None
g=None

def index_2d(data,element):
    for i in range(len(data)):
        if element in data[i]:
            return  i,data[i].index(element)

def hamming_distance(state,goal):
    no_of_misplaced_tiles=0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] and state[i][j]!=goal[i][j]:
                no_of_misplaced_tiles+=1
    return no_of_misplaced_tiles


def manhatten_distance(state,goal):
    dist=0
    for i in range(len(state)):
        for j in range(len(state[i])):
            goal_i,goal_j=index_2d(goal,state[i][j])
            dist+=abs(i-goal_i)+abs(j-goal_j)
    return dist



class Node:
    def __init__(self,data,goal_node,parent_node,level,goal=False):
        self.data=data #matrix representing position of tiles
        if goal:self.goal_node=self #pointer to self if self is goal node
        else:self.goal_node=goal_node #pointer to goal node
        self.h=None #heuristic value
        self.childs=[] #pointers to child nodes
        self.level=level #level of node in search tree
        self.parent=parent_node #pointer to parent node

    def calculate_h(self):
        self.discover() #call to discover for dicovering successors and appending them to childs data memeber
        if self.h==None:
            if dist_metric=='hamming':
                if algo=='a':self.h=hamming_distance(self.data,self.goal_node.data)+self.level
                elif algo=='b':self.h=hamming_distance(self.data,self.goal_node.data)
            elif dist_metric=='manhatten':
                if algo=='a':self.h=manhatten_distance(self.data,self.goal_node.data)+self.level
                elif algo=='b':self.h=manhatten_distance(self.data,self.goal_node.data)
        if dist_metric=='hamming':
            for child in self.childs:
                if algo=='a':child.h=hamming_distance(child.data,child.goal_node.data)+child.level
                elif algo=='b':child.h=hamming_distance(child.data,child.goal_node.data)
        elif dist_metric=='manhatten':
            for child in self.childs:
                if algo=='a':child.h=manhatten_distance(child.data,child.goal_node.data)+child.level
                elif algo=='b':child.h=manhatten_distance(child.data,child.goal_node.data)

    def discover(self):
        i,j=index_2d(self.data,None)
        possibility=[(i,j-1),(i,j+1),(i+1,j),(i-1,j)]
        for e in possibility:
            i1,j1=e[0],e[1]
            if i1 in range(len(self.data)) and  j1 in range(len(self.data)):
                child_data=copy.deepcopy(self.data)
                child_data[i][j]=self.data[i1][j1]
                child_data[i1][j1]=self.data[i][j]
                #print(child_data)
                self.childs.append(Node(child_data,self.goal_node,self,self.level+1))

def get_min_h_node():
    global open_list
    min_h_node=open_list[0]
    for node in open_list:
        node.calculate_h()
        if node.h<min_h_node.h:min_h_node=node
    return min_h_node


def is_node_less_than_same_level_nodes(current_node,list_type):
    global open_list
    global closed_list
    if list_type=='open': current_list=open_list
    elif list_type=='closed': current_list=closed_list
    for node in current_list:
        #current_node.level==node.level
        if current_node.level==node.level and current_node.h>node.h:
            return False
    return True

def is_member(node,list_type):
    if list_type=='open': current_list=open_list
    elif list_type=='closed': current_list=closed_list
    for e in current_list:
        if e.data==node.data:
            return True
    return False

def Bfs_Astar():
    global algo
    global dist_metric
    global g
    global open_list
    global closed_list
    found=False
    #inputs for type of algorithmand distance metric
    algo=input("A* or BFS[a/b] : ").lower()
    dist_metric=input("Manhatten or Hamming[m/h] : ").lower()
    if dist_metric=='m': dist_metric='manhatten'
    elif dist_metric=='h': dist_metric='hamming'
    #create start and goal node and add start node to openlist
    goal_node=Node(goal_state,None,None,None,True)
    start_node=Node(start_state,goal_node,None,0)
    open_list.append(start_node)
    # 
    while open_list:
        if found:break
        print('Content in open list')
        for e in open_list:
            print(e.data)
        print()
        current_node=get_min_h_node()
        open_list.remove(current_node)
        if is_member(current_node,'closed'):
            continue
        for child in current_node.childs:
            if child.data==goal_node.data:
                found=True
                print('Goal state Found!!!')
                break
            if not is_member(child,'open') and not is_member(child,'closed'):
                is_less_open=is_node_less_than_same_level_nodes(current_node,'open')
                is_less_closed=is_node_less_than_same_level_nodes(current_node,'closed')
                if is_less_open and is_less_closed:
                    open_list.append(child)
        closed_list.append(current_node)

Bfs_Astar()
'''
Node structure
[data,goal_node,parent_node,level,goal=False]

A* Search Algorithm
1.  Initialize the open list
2.  Initialize the closed list
    put the starting node on the open
    list (you can leave its f at zero)

3.  while the open list is not empty
    a) find the node with the least f on
       the open list, call it "q"

    b) pop q off the open list

    c) generate q's 8 successors and set their
       parents to q

    d) for each successor
        i) if successor is the goal, stop search
          successor.g = q.g + distance between
                              successor and q
          successor.h = distance from goal to
          successor (This can be done using many
          ways, we will discuss three heuristics-
          Manhattan, Diagonal and Euclidean
          Heuristics)

          successor.f = successor.g + successor.h

        ii) if a node with the same position as
            successor is in the OPEN list which has a
           lower f than successor, skip this successor

        iii) if a node with the same position as
            successor  is in the CLOSED list which has
            a lower f than successor, skip this successor
            otherwise, add  the node to the open list
     end (for loop)

    e) push q on the closed list
    end (while loop)
'''
