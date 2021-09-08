from tiles_inpt import *
import time
import copy
import random
import math

path_trace=[]
visited=[]
algo='b'
inpt=input("Manhatten or Hamming [m/h] :").lower()
if inpt=='m':dist_metric='manhatten'
elif inpt=='h':dist_metric='hamming'

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

    def _sorted(self):
        self.childs=sorted(self.childs,key=lambda x: x.h)


def sa():
    t=10**6
    tt=10**6
    cnt=0
    def cooling(t,cnt):
        #return tt/(1+math.log(1+cnt)) 
        return tt/cnt
        #return t-1
    print("*"*10,"Simulated Annealing","*"*10)
    start=time.time()
    goal_flag=False
    goal_node=Node(goal_state,None,None,None,True)
    start_node=Node(start_state,goal_node,None,0)
    current_node=start_node
    path_trace.append(current_node)
    while True:
        current_node.calculate_h()
        shuffled_child_no=list(range(len(current_node.childs)))
        random.shuffle(shuffled_child_no)
        for i in range(len(current_node.childs)):
            random_child_no=shuffled_child_no[i]
            delta_e=current_node.childs[random_child_no].h - current_node.h
            if delta_e < 0:
                current_node=current_node.childs[random_child_no]
                path_trace.append(current_node)
                if current_node.data==goal_node.data:goal_flag=True
                break
            probability=math.e**(-delta_e/t)
            r=random.random()
            if r<= probability:
                current_node=current_node.childs[random_child_no]
                path_trace.append(current_node)
                if current_node.data==goal_node.data:goal_flag=True
                break
        if goal_flag:
            break
        cnt+=1
        t=cooling(t,cnt)
        if cnt==1500 or t==0:
            print("Cnt",cnt)
            print("t",t)
            break
    stop=time.time()
    print("Temperature :",tt)
    print("start state :",start_node.data)
    print("goal state :",goal_node.data)
    print("current node where program halted :",current_node.data,"h value :",current_node.h)
    print("Time :",stop-start)
    if current_node.data==goal_node.data:
        print("Sucess")
        for e in path_trace:
            print(e.data)
    else:print("failure")
    print("Heuristic :",dist_metric)



def hill_climb(flavour):
    print("*"*10,"Hill Climbing","*"*10)
    start=time.time()
    goal_node=Node(goal_state,None,None,None,True)
    start_node=Node(start_state,goal_node,None,0)
    current_node=start_node
    path_trace.append(current_node)
    while True:
        next_min=False
        current_node.calculate_h()
        visited.append(copy.deepcopy(current_node.data))
        if flavour=="steepest_ascent":
            current_node._sorted()
        for child in current_node.childs:
            if child.h<=current_node.h and child.data not in visited:
                print("Next min h found")
                #print(child.data)
                next_min=True
                current_node=child
                path_trace.append(current_node)
                break
        if not next_min:print("No next min h!!!");break
    stop=time.time()
    print("Type :",flavour)
    print("start state :",start_node.data)
    print("goal state :",goal_node.data)
    print("current node where program halted :",current_node.data,"h value :",current_node.h)
    print("Time :",stop-start)
    print("Heuristic :",dist_metric)

inpt=input("Simualted annealing or Hill climbing[S/H] :").lower()
if inpt=='s':
    sa()
elif inpt=='h':
    hill_climb("simple")
    hill_climb("steepest_ascent")
