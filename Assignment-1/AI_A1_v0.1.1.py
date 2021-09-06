from tiles_inpt import *
import time
from pprint import pprint
import copy

algo=input("A* or Bfs[a/b] : ").lower()
metric=input("Manhatten or Hamming[m/h] : ").lower()
if algo=='a':
    g=1
elif algo=='b':
    g=0
if metric=='m':
    type='manhatten'
elif metric=='h':
    type='hamming'
visited=[]
path_trace=[]
stack=[]
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
    def __init__(self,d,goal_node,parent_node,level,goal=False):
        self.data=d
        if goal:
            self.goal_node=self
        else:
            self.goal_node=goal_node
        self.h=None
        self.childs=[]
        self.level=level
        self.parent=parent_node

    def calculate_h(self):
        self.discover()
        if self.h==None:
            if type=='hamming':
                self.h=hamming_distance(self.data,self.goal_node.data)+self.level
            elif type=='manhatten':
                self.h=manhatten_distance(self.data,self.goal_node.data)+self.level
        if type=='hamming':
            for child in self.childs:
                child.h=hamming_distance(child.data,child.goal_node.data)+child.level
        elif type=='manhatten':
            for child in self.childs:
                child.h=manhatten_distance(child.data,child.goal_node.data)+child.level

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
                self.childs.append(Node(child_data,self.goal_node,self,self.level+g))

    def get_min_h_nodes_same_parent(self):
        if self.data not in visited:
            visited.append(copy.deepcopy(self.data))
        self.calculate_h()
        ret=[]
        possible_h=[]
        #print("childs")
        for  e in self.childs:
            #print(e.data,e.h)
            possible_h.append(e.h)
        #print()
        #just append childs in sorted order in stack for backtracking purpose,till while loop
        possible_h.sort()
        sorted_childs=[]
        for h in possible_h:
            for child in self.childs:
                if h==child.h and child not in sorted_childs and child.data not in visited:
                    sorted_childs.append(child)
        stack.append(sorted_childs)
        while True:
            #possible_h.sort()
            min_h=min(possible_h)
            #print("filter of visited and min")
            #for min_h in possible_h:
            for e in self.childs:
                if e.h==min_h and e.data not in visited:
                    #print(e.data,e.h)
                    ret.append(e)
            #print()
            if len(ret)!=0:
                break
            else:
                possible_h=list(filter(lambda x: x != min_h,possible_h))
                if len(possible_h)==0:
                    break
        for e in ret:
            visited.append(copy.deepcopy(e.data))
        return ret

    def get_trace(self,parent):
        current=self
        trace=[]
        while current!=parent:
            if current=='':
                print("Parent of start")
                break
            trace.append(current)
            current=current.parent
        #print('get_trace')
        #for e in trace[::-1]:
        #    print(e.data)
        return trace[::-1]



def get_min_h_different_parents(h_nodes):
    h_val=[]
    ret=[]
    for e in h_nodes:
        h_val.append(e.h)
    while True:
        min_h=min(h_val)
        for e in h_nodes:
            if e.h==min_h:
                ret.append(e)
        if len(ret)!=0:
            break
        else:
            h_value=list(filter(lambda x: x != min_h,h_value))
            if len(h_value)==0:
                break
    return ret


def goto_next_level_get_min(same_min_h_nodes_same_parent):
    h_nodes=[]
    for e in same_min_h_nodes_same_parent:
        h_nodes.extend(e.get_min_h_nodes_same_parent())
    #print("childrens of different parents")
    #for e in h_nodes:
    #    print(e.data)
    #print()
    if len(h_nodes)==0:
        return 0
    same_min_h_nodes_different_parents=get_min_h_different_parents(h_nodes)
    #print("finding min_h from childs with different parents")
    #for e in same_min_h_nodes_different_parents:
    #    print(e.data)
    #print()
    if len(same_min_h_nodes_different_parents)>1:
        return goto_next_level_get_min(same_min_h_nodes_different_parents)
    elif len(same_min_h_nodes_different_parents)==1:
        #print("inside len(same_min_h_nodes_different_parents==1)")
        #for e in same_min_h_nodes_different_parents:
        #    print(e.data)
        #print()
        return same_min_h_nodes_different_parents[0]

start=time.time()
goal_node=Node(goal_state,'','',9999999999,goal=True)
start_node=Node(start_state,goal_node,'',0)
current_node=start_node
path_trace.append(current_node)
while True:
    if current_node.h:
        if current_node.h-current_node.level in (0,1):
            break
    '''same_min_h_nodes_same_parent=current_node.get_min_h_nodes_same_parent()
    for e in same_min_h_nodes_same_parent:
    print(e.data)
    print()
    current_node=same_min_h_nodes_same_parent[0]
    same_min_h_nodes_same_parent=current_node.get_min_h_nodes_same_parent()
    for e in same_min_h_nodes_same_parent:
    print(e.data)
    print()
    next_node=goto_next_level_get_min(same_min_h_nodes_same_parent)
    '''
    #print(same_min_h_nodes_same_parent[0].data)
    same_min_h_nodes_same_parent=current_node.get_min_h_nodes_same_parent()
    #print("child with same parent")
    #for e in same_min_h_nodes_same_parent:
    #    print(e.data)
    #print()
    if len(same_min_h_nodes_same_parent)>1:
        #print("goto next level get min")
        next_node=goto_next_level_get_min(same_min_h_nodes_same_parent)
        if next_node==0:
            path_trace=path_trace[:-1]
            bt=False
            for e in range(len(stack)-1,-1,-1):
                for i in range(len(stack[e])):
                    if stack[e][i]==same_min_h_nodes_same_parent[-1]:
                        if i==len(stack[e])-1:
                            next_node=stack[e-1][0]
                        else:next_node=stack[e][i+1]
                        visited.append(copy.deepcopy(next_node.data))
                        print("Found on stack")
                        bt=True
                        #path_trace.append(current_node)
                        break
                if bt:break
        #print("return from goto next level...")
        #print("next node is...")
        #print(next_node.data)
        #print()
        sub_path=next_node.get_trace(path_trace[-1])
        #print("sub path")
        #for e in sub_path:
        #    print(e.data)
        #print()
        current_node=next_node
        path_trace.extend(sub_path)
        #elif len(same_min_h_nodes_same_parent)==1:
    else:
        #len(same_min_h_nodes_same_parent)==1:
        if len(same_min_h_nodes_same_parent)==0:
            print("inside length 0")
            print(current_node.data)
            path_trace=path_trace[:-1]
            bt=False
            for e in range(len(stack)-1,-1,-1):
                for i in range(len(stack[e])):
                    if stack[e][i]==current_node:
                        if i==len(stack[e])-1:
                            current_node=stack[e-1][0]
                        else:current_node=stack[e][i+1]
                        visited.append(copy.deepcopy(current_node.data))
                        print("Found on stack")
                        bt=True
                        path_trace.append(current_node)
                        break
                if bt:break
        if len(same_min_h_nodes_same_parent)==1:
            current_node=same_min_h_nodes_same_parent[0]
            path_trace.append(current_node)

stop=time.time()
if current_node.data==goal_node.data:
    print("SUCCESS !!!")
    print("[ start state ]")
    print(start_node.data)
    print("[ goal state ]")
    print(goal_node.data)
    print("Total no of states explored",len(visited))
    print("[ optimal path ]")
    for e in path_trace:
        print(e.data)
    print("optimal path cost considering bot start and goal state",len(path_trace))
    print("Time Taken",stop-start)
else:
    print("Failure !!!")
    print("[ start state ]")
    print(start_node.data)
    print("[ goal state ]")
    print(goal_node.data)
    print("Current node where program halted")
    print(current_node.data)
    print("Total no of states explored",len(visited))
    print("path cost",len(path_trace))
    print("Time Taken",stop-start)

#c=current_node
#p_t=[]
#while c!=start_node:
#    p_t.append(c)
#    c=c.parent
 
 
